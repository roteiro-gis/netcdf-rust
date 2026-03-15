use std::sync::Arc;

use crate::attribute_api::Attribute;
use crate::btree_v1;
use crate::cache::ChunkCache;
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::io::Cursor;
use crate::local_heap::LocalHeap;
use crate::messages::link::{LinkMessage, LinkTarget};
use crate::messages::link_info::LinkInfoMessage;
use crate::messages::symbol_table_msg::SymbolTableMessage;
use crate::messages::HdfMessage;
use crate::object_header::ObjectHeader;
use crate::symbol_table::SymbolTableNode;

/// A group within an HDF5 file.
pub struct Group<'f> {
    file_data: &'f [u8],
    offset_size: u8,
    length_size: u8,
    pub(crate) name: String,
    pub(crate) address: u64,
    pub(crate) chunk_cache: Arc<ChunkCache>,
    /// Cached children: (name, object_header_address, is_group)
    children: Option<Vec<ChildEntry>>,
}

#[derive(Debug, Clone)]
struct ChildEntry {
    name: String,
    address: u64,
}

impl<'f> Group<'f> {
    /// Create a group from a known object header address.
    pub(crate) fn new(
        file_data: &'f [u8],
        address: u64,
        name: String,
        offset_size: u8,
        length_size: u8,
        chunk_cache: Arc<ChunkCache>,
    ) -> Self {
        Group {
            file_data,
            offset_size,
            length_size,
            name,
            address,
            chunk_cache,
            children: None,
        }
    }

    /// Group name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// List all child groups.
    pub fn groups(&self) -> Result<Vec<Group<'f>>> {
        let children = self.resolve_children()?;
        let mut groups = Vec::new();
        for child in &children {
            if self.is_group_at(child.address)? {
                groups.push(Group::new(
                    self.file_data,
                    child.address,
                    child.name.clone(),
                    self.offset_size,
                    self.length_size,
                    self.chunk_cache.clone(),
                ));
            }
        }
        Ok(groups)
    }

    /// Get a child group by name.
    pub fn group(&self, name: &str) -> Result<Group<'f>> {
        let children = self.resolve_children()?;
        for child in &children {
            if child.name == name {
                if self.is_group_at(child.address)? {
                    return Ok(Group::new(
                        self.file_data,
                        child.address,
                        child.name.clone(),
                        self.offset_size,
                        self.length_size,
                        self.chunk_cache.clone(),
                    ));
                } else {
                    return Err(Error::GroupNotFound(format!(
                        "'{}' is a dataset, not a group",
                        name
                    )));
                }
            }
        }
        Err(Error::GroupNotFound(name.to_string()))
    }

    /// List all child datasets.
    pub fn datasets(&self) -> Result<Vec<Dataset<'f>>> {
        let children = self.resolve_children()?;
        let mut datasets = Vec::new();
        for child in &children {
            if !self.is_group_at(child.address)? {
                datasets.push(Dataset::from_object_header(
                    self.file_data,
                    child.address,
                    child.name.clone(),
                    self.offset_size,
                    self.length_size,
                    self.chunk_cache.clone(),
                )?);
            }
        }
        Ok(datasets)
    }

    /// Get a child dataset by name.
    pub fn dataset(&self, name: &str) -> Result<Dataset<'f>> {
        let children = self.resolve_children()?;
        for child in &children {
            if child.name == name {
                return Dataset::from_object_header(
                    self.file_data,
                    child.address,
                    child.name.clone(),
                    self.offset_size,
                    self.length_size,
                    self.chunk_cache.clone(),
                );
            }
        }
        Err(Error::DatasetNotFound(name.to_string()))
    }

    /// List attributes on this group.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        let header = ObjectHeader::parse_at(
            self.file_data,
            self.address,
            self.offset_size,
            self.length_size,
        )?;
        let mut attrs = Vec::new();
        for msg in header.messages {
            if let HdfMessage::Attribute(attr) = msg {
                attrs.push(Attribute::from_message(attr));
            }
        }
        Ok(attrs)
    }

    /// Find an attribute by name.
    pub fn attribute(&self, name: &str) -> Result<Attribute> {
        let attrs = self.attributes()?;
        attrs
            .into_iter()
            .find(|a| a.name == name)
            .ok_or_else(|| Error::AttributeNotFound(name.to_string()))
    }

    /// Resolve children from the object header.
    /// Handles both old-style (symbol table) and new-style (link messages) groups.
    fn resolve_children(&self) -> Result<Vec<ChildEntry>> {
        if let Some(ref children) = self.children {
            return Ok(children.clone());
        }

        let header = ObjectHeader::parse_at(
            self.file_data,
            self.address,
            self.offset_size,
            self.length_size,
        )?;

        let mut children = Vec::new();

        // Check for old-style groups (symbol table message)
        let mut found_symbol_table = false;
        // Check for new-style groups (link messages)
        let mut link_info: Option<LinkInfoMessage> = None;
        let mut links: Vec<LinkMessage> = Vec::new();

        for msg in &header.messages {
            match msg {
                HdfMessage::SymbolTable(st) => {
                    found_symbol_table = true;
                    children = self.resolve_old_style_group(st)?;
                }
                HdfMessage::Link(link) => {
                    links.push(link.clone());
                }
                HdfMessage::LinkInfo(li) => {
                    link_info = Some(li.clone());
                }
                _ => {}
            }
        }

        if !found_symbol_table {
            // New-style group: use compact links from header messages
            for link in &links {
                match &link.target {
                    LinkTarget::Hard { address } => {
                        children.push(ChildEntry {
                            name: link.name.clone(),
                            address: *address,
                        });
                    }
                    LinkTarget::Soft { .. } | LinkTarget::External { .. } => {
                        // Skip soft and external links for now
                    }
                }
            }

            // If there's a link_info pointing to a fractal heap, resolve dense links
            if children.is_empty() {
                if let Some(ref li) = link_info {
                    if !Cursor::is_undefined_offset(li.fractal_heap_address, self.offset_size) {
                        children = self.resolve_dense_links(li)?;
                    }
                }
            }
        }

        Ok(children)
    }

    /// Resolve old-style group children via B-tree v1 + local heap.
    fn resolve_old_style_group(&self, st: &SymbolTableMessage) -> Result<Vec<ChildEntry>> {
        // Parse the local heap to get the name table
        let mut heap_cursor = Cursor::new(self.file_data);
        heap_cursor.set_position(st.heap_address);
        let heap = LocalHeap::parse(&mut heap_cursor, self.offset_size, self.length_size)?;

        // Walk the B-tree to collect all symbol table node addresses
        let leaves = btree_v1::collect_btree_v1_leaves(
            self.file_data,
            st.btree_address,
            self.offset_size,
            self.length_size,
            None, // group B-tree, no ndims
        )?;

        let mut children = Vec::new();

        for (_key, snod_address) in &leaves {
            let mut cursor = Cursor::new(self.file_data);
            cursor.set_position(*snod_address);
            let snod = SymbolTableNode::parse(&mut cursor, self.offset_size, self.length_size)?;

            for entry in &snod.entries {
                let name = heap.get_string(entry.link_name_offset, self.file_data)?;
                children.push(ChildEntry {
                    name,
                    address: entry.object_header_address,
                });
            }
        }

        Ok(children)
    }

    /// Resolve dense links from a fractal heap + B-tree v2.
    /// This is the Phase 2 path — requires fractal heap implementation.
    fn resolve_dense_links(&self, _link_info: &LinkInfoMessage) -> Result<Vec<ChildEntry>> {
        // TODO: Phase 2 — implement fractal heap traversal for dense links
        Ok(Vec::new())
    }

    /// Check if the object at the given address is a group (vs a dataset).
    /// A group has either a symbol table message, link messages, or link info.
    /// A dataset has a dataspace + datatype + layout.
    fn is_group_at(&self, address: u64) -> Result<bool> {
        let header =
            ObjectHeader::parse_at(self.file_data, address, self.offset_size, self.length_size)?;
        for msg in &header.messages {
            match msg {
                // Group indicators
                HdfMessage::SymbolTable(_)
                | HdfMessage::Link(_)
                | HdfMessage::LinkInfo(_)
                | HdfMessage::GroupInfo(_) => return Ok(true),
                // Dataset indicators
                HdfMessage::DataLayout(_) => return Ok(false),
                _ => {}
            }
        }
        // Default: if it has neither, treat as group (root groups can be empty)
        Ok(true)
    }
}
