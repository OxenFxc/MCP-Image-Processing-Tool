# MCP Image Processing Tool | MCP图片处理工具

[English](#english) | [中文](#中文)

<div id="english">

## Overview

This is an MCP (Model Context Protocol) based image processing tool that supports conversion between arrays and images, as well as chunked processing for large images.

### Key Features

- ✅ **Array to Image**: Convert 3D arrays to base64 encoded images
- ✅ **Image to Array**: Convert base64 encoded images to 3D arrays
- ✅ **Chunked Processing**: Support large image compression and parsing
- ✅ **Multiple Formats**: Support PNG, JPEG, BMP and other formats
- ✅ **Multi-channel Support**: Support Grayscale(1ch), RGB(3ch), RGBA(4ch)
- ✅ **Example Generation**: Built-in pattern generators for testing

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

1. Start MCP Server:
```bash
python mcp_image_server.py
```

2. Run Test Example:
```bash
python test_example.py
```

### Available Tools

#### `array_to_image`
Convert 3D array to base64 encoded image

**Parameters**:
- `data`: 3D array [height, width, channels], value range 0-255
- `format`: Image format (PNG, JPEG, BMP etc.), default PNG

**Example**:
```json
{
  "data": [
    [[255, 0, 0], [0, 255, 0]],
    [[0, 0, 255], [255, 255, 255]]
  ],
  "format": "PNG"
}
```

#### `image_to_array`
Convert base64 encoded image to 3D array

**Parameters**:
- `image_base64`: Base64 encoded image string

#### `create_chunked_image`
Create chunked image data for large images

**Parameters**:
- `data`: Large image 3D array
- `format`: Image format, default PNG
- `chunk_size`: Chunk size in bytes, default 1MB

#### `parse_chunked_image`
Parse chunked image data back to complete array

**Parameters**:
- `chunked_data`: Chunked data dictionary

#### `get_image_info`
Get basic image information

**Parameters**:
- `image_base64`: Base64 encoded image string

#### `create_example_array`
Create example array for testing

**Parameters**:
- `width`: Image width, default 100
- `height`: Image height, default 100
- `pattern`: Pattern type (gradient, checkerboard, solid, random), default gradient
- `channels`: Number of channels (1=grayscale, 3=RGB, 4=RGBA), default 3

#### `file_to_array`
Read image file and convert to 3D array

**Parameters**:
- `file_path`: Path to the image file (supports both relative and absolute paths)

#### `save_base64_to_file`
Save base64 encoded image to file

**Parameters**:
- `image_base64`: Base64 encoded image string
- `file_path`: Path to save the image file (supports both relative and absolute paths)
- `format`: Image format (optional, will be inferred from file extension if not specified)

#### `save_array_to_file`
Save base64 encoded image to file

**Parameters**:
- `image_base64`: Base64 encoded image string
- `file_path`: Path to save the image file (supports both relative and absolute paths)
- `format`: Image format (optional, will be inferred from file extension if not specified)

#### `save_array`
Save array data directly to file

**Parameters**:
- `array_data`: 3D array [height, width, channels]
- `file_path`: Path to save the array file
- `format`: Save format ("npy" or "json"), default "npy"

**Example**:
```python
# Save array data
array_data = processor.image_to_array(image_base4)
# Save as numpy format (more efficient)
processor.save_array(array_data, "output/data.npy")
# Or save as JSON format (human readable)
processor.save_array(array_data, "output/data.json", format="json")
```

#### `load_array`
Load array data from file

**Parameters**:
- `file_path`: Path to the array data file (.npy or .json)

**Example**:
```python
# Load from numpy format
array_data = processor.load_array("output/data.npy")
# Or load from JSON format
array_data = processor.load_array("output/data.json")
```

### Use Cases

1. Basic Image Processing
```python
# AI input array to create image
array_data = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
# Use array_to_image tool to convert

# Parse image to array
# Use image_to_array tool to parse
```

2. Large Image Processing
```python
# For large images, use chunked processing to avoid memory overflow
large_array = create_large_image_array()
# Use create_chunked_image tool to create chunks
# Use parse_chunked_image tool to reconstruct
```

3. Image Information
```python
# Get detailed image information
# Use get_image_info tool to view width, height, format etc.
```

4. Array Data Storage
```python
# Convert image to array and save
array_data = processor.image_to_array(image_base64)
# Save as numpy format for efficiency
processor.save_array(array_data, "arrays/image_data.npy")
# Or save as JSON for readability
processor.save_array(array_data, "arrays/image_data.json", format="json")

# Load array data later
array_data = processor.load_array("arrays/image_data.npy")
# Process or convert back to image
image_base64 = processor.array_to_image(array_data)
```

### Configuration

#### MCP Config (config.json)
```json
{
  "mcpServers": {
    "image-processor": {
      "command": "python",
      "args": ["mcp_image_server.py"],
      "env": {}
    }
  }
}
```

#### Chunk Configuration
- Default chunk size: 1MB
- Maximum image size: 2048x2048
- Customizable chunk size

### Technical Details

#### Array Format
- **2D Array**: [height, width] - Grayscale
- **3D Array**: [height, width, channels] - Color
- **Value Range**: 0-255 (uint8)

#### Chunking Algorithm
1. Check if image size exceeds chunk limit
2. Split image into chunks by rows
3. Encode each chunk to base64
4. Save chunk info for reconstruction

#### Supported Formats
- PNG (Recommended, lossless)
- JPEG (Lossy compression)
- BMP (Uncompressed)
- Other PIL supported formats

### Error Handling
- Array dimension validation
- Value range checking
- Format validation
- Memory usage monitoring

### Performance Optimization
- Efficient array processing with numpy
- Smart chunking to avoid memory overflow
- Optimized base64 encoding
- Multiple compression format support

### License
MIT License

---

</div>

<div id="中文">

## 概述

这是一个基于MCP（Model Context Protocol）的图片处理工具，支持数组和图片之间的转换，以及大图片的分卷处理。

### 功能特性

- ✅ **数组转图片**: 将3D数组转换为base64编码的图片
- ✅ **图片转数组**: 将base64编码的图片转换为3D数组  
- ✅ **分卷处理**: 支持大图片的分卷压缩和解析
- ✅ **多种格式**: 支持PNG、JPEG、BMP等图片格式
- ✅ **多通道支持**: 支持灰度(1通道)、RGB(3通道)、RGBA(4通道)
- ✅ **示例生成**: 内置多种图案生成器用于测试

### 安装依赖

```bash
pip install -r requirements.txt
```

### 快速开始

1. 启动MCP服务器：
```bash
python mcp_image_server.py
```

2. 运行测试示例：
```bash
python test_example.py
```

### 可用工具

#### `array_to_image`
将3D数组转换为base64编码的图片

**参数**:
- `data`: 3D数组 [height, width, channels]，值范围0-255
- `format`: 图片格式 (PNG, JPEG, BMP等)，默认PNG

**示例**:
```json
{
  "data": [
    [[255, 0, 0], [0, 255, 0]],
    [[0, 0, 255], [255, 255, 255]]
  ],
  "format": "PNG"
}
```

#### `image_to_array`
将base64编码的图片转换为3D数组

**参数**:
- `image_base64`: base64编码的图片字符串

#### `create_chunked_image`
创建分卷图片数据，用于处理大图片

**参数**:
- `data`: 大图片的3D数组
- `format`: 图片格式，默认PNG
- `chunk_size`: 分块大小（字节），默认1MB

#### `parse_chunked_image`
解析分卷图片数据，重组为完整图片数组

**参数**:
- `chunked_data`: 分卷数据字典

#### `get_image_info`
获取图片的基本信息

**参数**:
- `image_base64`: base64编码的图片字符串

#### `create_example_array`
创建示例数组，用于测试图片生成

**参数**:
- `width`: 图片宽度，默认100
- `height`: 图片高度，默认100
- `pattern`: 图案类型 (gradient, checkerboard, solid, random)，默认gradient
- `channels`: 通道数 (1=灰度, 3=RGB, 4=RGBA)，默认3

#### `file_to_array`
读取图片文件并转换为3D数组

**参数**:
- `file_path`: 图片文件的路径（支持相对路径和绝对路径）

#### `save_base64_to_file`
将base64编码的图片保存为文件

**参数**:
- `image_base64`: base64编码的图片字符串
- `file_path`: 保存的文件路径（支持相对路径和绝对路径）
- `format`: 图片格式（可选，如果不指定则从文件扩展名推断）

#### `save_array_to_file`
将base64编码的图片保存为文件

**参数**:
- `image_base64`: base64编码的图片字符串
- `file_path`: 保存的文件路径（支持相对路径和绝对路径）
- `format`: 图片格式（可选，如果不指定则从文件扩展名推断）

#### `save_array`
直接保存数组数据到文件

**参数**:
- `array_data`: 3D数组 [height, width, channels]
- `file_path`: 保存的文件路径
- `format`: 保存格式（"npy"或"json"），默认"npy"

**示例**:
```python
# 保存数组数据
array_data = processor.image_to_array(image_base64)
# 保存为numpy格式（更高效）
processor.save_array(array_data, "output/data.npy")
# 或保存为JSON格式（可读性好）
processor.save_array(array_data, "output/data.json", format="json")
```

#### `load_array`
从文件加载数组数据

**参数**:
- `file_path`: 数组数据文件路径（.npy或.json）

**示例**:
```python
# 从numpy格式加载
array_data = processor.load_array("output/data.npy")
# 或从JSON格式加载
array_data = processor.load_array("output/data.json")
```

### 使用场景

1. 基本图片处理
```python
# AI输入数组创建图片
array_data = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
# 使用 array_to_image 工具转换为图片

# 解析图片为数组
# 使用 image_to_array 工具解析图片
```

2. 大图片分卷处理
```python
# 对于大图片，使用分卷处理避免内存过载
large_array = create_large_image_array()
# 使用 create_chunked_image 工具创建分卷
# 使用 parse_chunked_image 工具重组图片
```

3. 图片信息查看
```python
# 获取图片详细信息
# 使用 get_image_info 工具查看宽度、高度、格式等
```

4. 数组数据存储
```python
# 将图片转换为数组并保存
array_data = processor.image_to_array(image_base64)
# 保存为numpy格式以提高效率
processor.save_array(array_data, "arrays/image_data.npy")
# 或保存为JSON格式以提高可读性
processor.save_array(array_data, "arrays/image_data.json", format="json")

# 之后加载数组数据
array_data = processor.load_array("arrays/image_data.npy")
# 处理或转换回图片
image_base64 = processor.array_to_image(array_data)
```

### 配置说明

#### MCP配置 (config.json)
```json
{
  "mcpServers": {
    "image-processor": {
      "command": "python",
      "args": ["mcp_image_server.py"],
      "env": {}
    }
  }
}
```

#### 分卷配置
- 默认分块大小: 1MB
- 最大图片尺寸: 2048x2048
- 支持自定义分块大小

### 技术细节

#### 数组格式
- **2D数组**: [height, width] - 灰度图
- **3D数组**: [height, width, channels] - 彩色图
- **数值范围**: 0-255 (uint8)

#### 分卷算法
1. 检查图片大小是否超过分块限制
2. 按行分割图片为多个块
3. 每个块独立编码为base64
4. 保存分块信息用于重组

#### 支持格式
- PNG (推荐，无损)
- JPEG (有损压缩)
- BMP (无压缩)
- 其他PIL支持的格式

### 错误处理
- 数组维度检查
- 数值范围验证
- 图片格式验证
- 内存使用监控

### 性能优化
- 使用numpy进行高效数组处理
- 智能分块避免内存溢出
- base64编码优化
- 支持多种压缩格式

### 许可证
MIT License

</div> 