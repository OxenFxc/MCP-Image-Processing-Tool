#!/usr/bin/env python3
"""
MCP图片处理工具服务器
支持数组和图片之间的转换，以及大图片的分卷处理
"""

import asyncio
import json
import sys
from typing import Any, Dict, List
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from image_processor import ImageProcessor

# 创建MCP服务器实例
server = Server("image-processor")

# 创建图片处理器实例
image_processor = ImageProcessor()

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """返回可用的工具列表"""
    return [
        types.Tool(
            name="array_to_image",
            description="将3D数组转换为base64编码的图片",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "3D数组 [height, width, channels]，值范围0-255",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0, "maximum": 255}
                            }
                        }
                    },
                    "format": {
                        "type": "string",
                        "description": "图片格式 (PNG, JPEG, BMP等)",
                        "default": "PNG"
                    }
                },
                "required": ["data"]
            },
        ),
        types.Tool(
            name="image_to_array",
            description="将base64编码的图片转换为3D数组",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "base64编码的图片字符串"
                    }
                },
                "required": ["image_base64"]
            },
        ),
        types.Tool(
            name="create_chunked_image",
            description="创建分卷图片数据，用于处理大图片",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "大图片的3D数组 [height, width, channels]",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0, "maximum": 255}
                            }
                        }
                    },
                    "format": {
                        "type": "string",
                        "description": "图片格式 (PNG, JPEG等)",
                        "default": "PNG"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "分块大小（字节）",
                        "default": 1048576
                    }
                },
                "required": ["data"]
            },
        ),
        types.Tool(
            name="parse_chunked_image",
            description="解析分卷图片数据，重组为完整图片数组",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunked_data": {
                        "type": "object",
                        "description": "分卷数据字典",
                        "properties": {
                            "is_chunked": {"type": "boolean"},
                            "chunks": {"type": "array", "items": {"type": "string"}},
                            "chunk_info": {"type": "array"},
                            "original_shape": {"type": "array"},
                            "total_chunks": {"type": "integer"},
                            "format": {"type": "string"},
                            "data": {"type": "string"}
                        }
                    }
                },
                "required": ["chunked_data"]
            },
        ),
        types.Tool(
            name="get_image_info",
            description="获取图片的基本信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "base64编码的图片字符串"
                    }
                },
                "required": ["image_base64"]
            },
        ),
        types.Tool(
            name="create_example_array",
            description="创建示例数组，用于测试图片生成",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "description": "图片宽度",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 2048
                    },
                    "height": {
                        "type": "integer",
                        "description": "图片高度",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 2048
                    },
                    "pattern": {
                        "type": "string",
                        "description": "图案类型 (gradient, checkerboard, solid, random)",
                        "default": "gradient"
                    },
                    "channels": {
                        "type": "integer",
                        "description": "通道数 (1=灰度, 3=RGB, 4=RGBA)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 4
                    }
                },
                "required": []
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """处理工具调用"""
    
    try:
        if name == "array_to_image":
            data = arguments["data"]
            format_type = arguments.get("format", "PNG")
            
            result = image_processor.array_to_image(data, format_type)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"成功将数组转换为{format_type}格式的图片。\n"
                         f"图片数据 (base64): {result[:100]}..."
                )
            ]
            
        elif name == "image_to_array":
            image_base64 = arguments["image_base64"]
            
            result = image_processor.image_to_array(image_base64)
            
            # 获取数组信息
            import numpy as np
            np_array = np.array(result)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"成功将图片转换为数组。\n"
                         f"数组形状: {np_array.shape}\n"
                         f"数值范围: {np_array.min()} - {np_array.max()}\n"
                         f"数组数据: {json.dumps(result, indent=2)}"
                )
            ]
            
        elif name == "create_chunked_image":
            data = arguments["data"]
            format_type = arguments.get("format", "PNG")
            chunk_size = arguments.get("chunk_size", 1048576)
            
            # 临时设置分块大小
            original_chunk_size = image_processor.chunk_size
            image_processor.chunk_size = chunk_size
            
            try:
                result = image_processor.create_chunked_image(data, format_type)
            finally:
                image_processor.chunk_size = original_chunk_size
            
            if result["is_chunked"]:
                return [
                    types.TextContent(
                        type="text",
                        text=f"成功创建分卷图片数据。\n"
                             f"原始形状: {result['original_shape']}\n"
                             f"分块数量: {result['total_chunks']}\n"
                             f"格式: {result['format']}\n"
                             f"分卷数据: {json.dumps(result, indent=2)}"
                    )
                ]
            else:
                return [
                    types.TextContent(
                        type="text",
                        text=f"图片较小，无需分卷。\n"
                             f"原始形状: {result['original_shape']}\n"
                             f"格式: {result['format']}\n"
                             f"图片数据 (base64): {result['data'][:100]}..."
                    )
                ]
                
        elif name == "parse_chunked_image":
            chunked_data = arguments["chunked_data"]
            
            result = image_processor.parse_chunked_image(chunked_data)
            
            # 获取数组信息
            import numpy as np
            np_array = np.array(result)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"成功解析分卷图片数据。\n"
                         f"重组后数组形状: {np_array.shape}\n"
                         f"数值范围: {np_array.min()} - {np_array.max()}\n"
                         f"数组数据: {json.dumps(result, indent=2)}"
                )
            ]
            
        elif name == "get_image_info":
            image_base64 = arguments["image_base64"]
            
            result = image_processor.get_image_info(image_base64)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"图片信息:\n{json.dumps(result, indent=2)}"
                )
            ]
            
        elif name == "create_example_array":
            width = arguments.get("width", 100)
            height = arguments.get("height", 100)
            pattern = arguments.get("pattern", "gradient")
            channels = arguments.get("channels", 3)
            
            # 创建示例数组
            import numpy as np
            
            if pattern == "gradient":
                # 渐变图案
                if channels == 1:
                    array = np.zeros((height, width, 1), dtype=np.uint8)
                    for i in range(height):
                        array[i, :, 0] = int(255 * i / height)
                else:
                    array = np.zeros((height, width, channels), dtype=np.uint8)
                    for i in range(height):
                        for j in range(width):
                            array[i, j, 0] = int(255 * i / height)  # 红色渐变
                            if channels > 1:
                                array[i, j, 1] = int(255 * j / width)  # 绿色渐变
                            if channels > 2:
                                array[i, j, 2] = int(255 * (i + j) / (height + width))  # 蓝色渐变
                            if channels > 3:
                                array[i, j, 3] = 255  # Alpha通道
                                
            elif pattern == "checkerboard":
                # 棋盘图案
                array = np.zeros((height, width, channels), dtype=np.uint8)
                for i in range(height):
                    for j in range(width):
                        if (i // 10 + j // 10) % 2 == 0:
                            array[i, j] = 255
                            
            elif pattern == "solid":
                # 纯色
                array = np.full((height, width, channels), 128, dtype=np.uint8)
                
            elif pattern == "random":
                # 随机噪声
                array = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            
            else:
                raise ValueError(f"未知的图案类型: {pattern}")
            
            # 如果是单通道，移除最后一个维度
            if channels == 1:
                array = array.squeeze(axis=2)
            
            result = array.tolist()
            
            return [
                types.TextContent(
                    type="text",
                    text=f"成功创建示例数组。\n"
                         f"尺寸: {width}x{height}\n"
                         f"通道数: {channels}\n"
                         f"图案: {pattern}\n"
                         f"数组形状: {array.shape}\n"
                         f"数组数据: {json.dumps(result, indent=2)}"
                )
            ]
            
        else:
            raise ValueError(f"未知的工具: {name}")
            
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"工具执行失败: {str(e)}"
            )
        ]

async def main():
    """主函数"""
    # 运行MCP服务器
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="image-processor",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 