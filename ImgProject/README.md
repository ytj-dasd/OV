# PyIMSGui - File Editor with Sidebar Interface

A PySide6-based file editor application featuring a three-panel layout with collapsible sidebars, tabbed editing interface, and split-screen functionality.

## Project Overview

PyIMSGui implements a modern file editor interface similar to VS Code, featuring:
- Left sidebar: File browser with expandable file attributes
- Center: Tabbed editor with drag-and-drop split-screen support  
- Right sidebar: Auxiliary tools with dual-tab interface
- Collapsible sidebars with toggle buttons
- Custom txt file format parsing

## Architecture

### Modular Structure

```
pyIMSGui/
├── main.py                 # Main application entry point
├── src/
│   ├── __init__.py
│   ├── file_parser.py      # File format parsing
│   ├── widgets.py          # Core UI components
│   └── sidebar_widgets.py  # Sidebar components
├── example.txt            # Sample file for testing
├── pyproject.toml         # Project dependencies
└── README.md
```

### Core Components

#### 1. File Parser (`src/file_parser.py`)

**Purpose**: Handles parsing of custom txt file format with header/context sections.

**Format Specification**:
```
header
attribute1
attribute2
attribute3
header

context
content_line1
content_line2
context
```

**API**:
- `parse_file(file_path: str) -> Dict[str, List[str]]`
  - Returns: `{'header': [...], 'context': [...]}`

#### 2. Core Widgets (`src/widgets.py`)

##### ResizableSplitter
- Custom QSplitter with collapse functionality
- Double-click handles to toggle panel visibility
- Visual feedback for resize operations

##### SplitTabWidget
- Tabbed interface supporting multiple split views
- Key Features:
  - Drag-and-drop tab reordering
  - Context menu for splitting views
  - Automatic empty tab cleanup
  - File state tracking across splits

##### DraggableTabBar
- Custom QTabBar enabling tab dragging between splits
- Implements QDrag for smooth tab movement
- Visual feedback during drag operations

##### EditorWidget
- Main editing area container
- Integrates SplitTabWidget with context menus
- Handles file opening and tab management

#### 3. Sidebar Widgets (`src/sidebar_widgets.py`)

##### FileBrowserWidget (Left Sidebar)
**Functionality**:
- File tree display with expandable attributes
- Open/Delete/Collapse operations
- Double-click to open files in editor
- Synchronization with right sidebar

**UI Components**:
- Button toolbar (Open, Delete, Collapse)
- QTreeWidget for hierarchical file display
- File attribute expansion

**Methods**:
- `open_file()`: File dialog for txt file selection
- `add_file_to_tree()`: Populate tree with file data
- `delete_selected()`: Remove files from browser
- `on_file_double_click()`: Handle file opening

##### RightSidebarWidget
**Dual-Tab Interface**:

**Tab 1 - Files**: 
- File selector list synchronized with left sidebar
- Dual selection lists (Options/Values)
- Text display area
- Input box for notes

**Tab 2 - Selection**:
- Alternative file selector
- Attribute/value selection lists
- Detailed text display
- Additional input capabilities

#### 4. Main Application (`main.py`)

**MainWindow Class**:
- Orchestrates all components
- Manages three-panel layout with QSplitter
- Handles sidebar toggle functionality
- Provides menu bar integration

**Layout Structure**:
```
┌─────────────────────────────────────────────────────┐
│ Menu Bar                                            │
├─────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│ │ Left Panel  │ │ Editor Panel    │ │ Right Panel │ │
│ │ (250px)     │ │ (Expands)       │ │ (300px)     │ │
│ │             │ │                 │ │             │ │
│ │ FileBrowser │ │ SplitTabWidget  │ │ Tabs x2     │ │
│ │             │ │                 │ │             │ │
│ └─────────────┘ └─────────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Usage Flow

### 1. File Operations
1. **Open File**: File → Open File... or toolbar "Open" button
2. **File Display**: Files appear in left sidebar tree
3. **Attribute Expansion**: Click ▶/▼ to show/hide file attributes
4. **File Editing**: Double-click file to open in editor tab
5. **File Removal**: Select file → Delete button

### 2. Editor Operations
1. **Tab Management**: 
   - Open multiple files in separate tabs
   - Drag tabs between split panes
   - Close tabs with X button

2. **Split Screen**:
   - Right-click editor → "Split Right/Left"
   - Drag tabs to create new splits
   - Resize split panes via drag handles

### 3. Sidebar Management
- **Toggle Left**: ◀/▶ button or Ctrl+B
- **Toggle Right**: ▶/◀ button
- **Automatic Resize**: Sidebars collapse to minimum width

## File Format Details

### Input File Structure
The application expects txt files with specific format:

```
header
[attribute1]
[attribute2]
...
header

context
[content_line1]
[content_line2]
...
context
```

### Data Processing
- **Header Section**: Parsed into file attributes (displayed in tree)
- **Context Section**: Becomes file content (displayed in editor)
- **File Tracking**: Maintains open file state across sessions

## Technical Implementation

### Signal-Slot Architecture
- **FileBrowserWidget → MainWindow**: `open_file_in_editor()`
- **MainWindow → RightSidebarWidget**: `update_right_sidebar_files()`
- **File State**: Centralized in FileBrowserWidget.opened_files dict

### Drag and Drop System
- **DraggableTabBar**: Custom QTabBar implementation
- **QDrag Integration**: Uses QMimeData for tab data transfer
- **Visual Feedback**: Custom drag pixmap generation

### Layout Management
- **QSplitter**: Handles resizable three-panel layout
- **Stretch Factors**: Editor panel expands, sidebars maintain size
- **Collapse Animation**: Smooth transitions via QSplitter sizing

## Installation & Setup

### Prerequisites
- Python 3.11+
- PySide6 6.9.1+

### Installation
```bash
# Clone or extract project
cd pyIMSGui

# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### Running
```bash
uv run main.py
# or
python main.py
```

### Testing
Use the provided `example.txt` to test file operations:
```bash
# The file contains:
# header
# xyz
# intensity  
# rgb
# label
# header
#
# context
# random
# context
```

## Customization

### Styling
- Uses Qt Fusion style for cross-platform consistency
- CSS styling can be added via setStyleSheet()
- Fonts: Consolas 10pt for editor, system defaults for UI

### Extending
- Add new file formats by extending FileParser
- Implement additional sidebar tabs in RightSidebarWidget
- Add new editor features via EditorWidget extensions

## Known Limitations

- Display environment requires X11/Wayland (Linux GUI)
- Limited to txt format with specific structure
- No persistent settings storage
- Basic text editor functionality (no syntax highlighting)

## Future Enhancements

- Settings persistence (QSettings)
- Syntax highlighting for various formats
- File type detection and auto-parsing
- Advanced split-screen layouts
- Plugin system for extensibility