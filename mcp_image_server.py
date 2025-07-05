#!/usr/bin/env python3
"""
MCP图片处理工具服务器
支持数组和图片之间的转换，以及大图片的分卷处理
"""

import asyncio
import json
import sys
import os
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
            description="将3D数组转换为base64编码的图片，并可选择保存到文件",
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
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存的文件路径（可选，如果不提供则不保存文件）",
                        "default": None
                    }
                },
                "required": ["data"]
            },
        ),
        types.Tool(
            name="image_to_array",
            description="将base64编码的图片转换为3D数组，并可选择保存为JSON文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "base64编码的图片字符串"
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存的JSON文件路径（可选，如果不提供则不保存文件）",
                        "default": None
                    }
                },
                "required": ["image_base64"]
            },
        ),
        types.Tool(
            name="create_chunked_image",
            description="创建分卷图片数据，用于处理大图片，并可选择保存为JSON文件",
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
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存的JSON文件路径（可选，如果不提供则不保存文件）",
                        "default": None
                    }
                },
                "required": ["data"]
            },
        ),
        types.Tool(
            name="parse_chunked_image",
            description="解析分卷图片数据，重组为完整图片数组，并可选择保存为JSON文件",
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
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存的JSON文件路径（可选，如果不提供则不保存文件）",
                        "default": None
                    }
                },
                "required": ["chunked_data"]
            },
        ),
        types.Tool(
            name="get_image_info",
            description="获取图片的基本信息，并可选择保存为JSON文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "base64编码的图片字符串"
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存的JSON文件路径（可选，如果不提供则不保存文件）",
                        "default": None
                    }
                },
                "required": ["image_base64"]
            },
        ),
        types.Tool(
            name="create_example_array",
            description="创建示例数组，用于测试图片生成，并可选择保存为JSON文件",
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
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存的JSON文件路径（可选，如果不提供则不保存文件）",
                        "default": None
                    }
                },
                "required": []
            },
        ),
        types.Tool(
            name="file_to_array",
            description="从文件路径读取图片并转换为3D数组，并可选择保存为JSON文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "图片文件的路径（支持相对路径和绝对路径）"
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存的JSON文件路径（可选，如果不提供则不保存文件）",
                        "default": None
                    }
                },
                "required": ["file_path"]
            },
        ),
        types.Tool(
            name="save_base64_to_file",
            description="将base64编码的图片保存为文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "base64编码的图片字符串"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "保存的文件路径（支持相对路径和绝对路径）"
                    },
                    "format": {
                        "type": "string",
                        "description": "图片格式（可选，如果不指定则从文件扩展名推断）",
                        "default": None
                    }
                },
                "required": ["image_base64", "file_path"]
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
            save_path = arguments.get("save_path")
            
            result = image_processor.array_to_image(data, format_type, save=bool(save_path), filename=save_path)
            
            response_text = f"成功将数组转换为{format_type}格式的图片。\n"
            if save_path:
                response_text += f"图片已保存到: {save_path}\n"
            response_text += f"图片数据 (base64): {result[:100]}..."
            
            return [types.TextContent(type="text", text=response_text)]
            
        elif name == "image_to_array":
            image_base64 = arguments["image_base64"]
            save_path = arguments.get("save_path")
            
            result = image_processor.image_to_array(image_base64, save=bool(save_path), filename=save_path)
            
            # 获取数组信息
            import numpy as np
            np_array = np.array(result)
            
            response_text = f"成功将图片转换为数组。\n"
            if save_path:
                response_text += f"数组已保存到: {save_path}\n"
            response_text += f"数组形状: {np_array.shape}\n"
            response_text += f"数值范围: {np_array.min()} - {np_array.max()}\n"
            response_text += f"数组数据: {json.dumps(result, indent=2)}"
            
            return [types.TextContent(type="text", text=response_text)]
            
        elif name == "create_chunked_image":
            data = arguments["data"]
            format_type = arguments.get("format", "PNG")
            chunk_size = arguments.get("chunk_size", 1048576)
            save_path = arguments.get("save_path")
            
            # 临时设置分块大小
            original_chunk_size = image_processor.chunk_size
            image_processor.chunk_size = chunk_size
            
            try:
                result = image_processor.create_chunked_image(data, format_type, save=bool(save_path), filename=save_path)
            finally:
                image_processor.chunk_size = original_chunk_size
            
            response_text = ""
            if result["is_chunked"]:
                response_text = f"成功创建分卷图片数据。\n"
                if save_path:
                    response_text += f"分卷数据已保存到: {save_path}\n"
                response_text += f"原始形状: {result['original_shape']}\n"
                response_text += f"分块数量: {result['total_chunks']}\n"
                response_text += f"格式: {result['format']}\n"
                response_text += f"分卷数据: {json.dumps(result, indent=2)}"
            else:
                response_text = f"图片较小，无需分卷。\n"
                if save_path:
                    response_text += f"数据已保存到: {save_path}\n"
                response_text += f"原始形状: {result['original_shape']}\n"
                response_text += f"格式: {result['format']}\n"
                response_text += f"图片数据 (base64): {result['data'][:100]}..."
            
            return [types.TextContent(type="text", text=response_text)]
                
        elif name == "parse_chunked_image":
            chunked_data = arguments["chunked_data"]
            save_path = arguments.get("save_path")
            
            result = image_processor.parse_chunked_image(chunked_data, save=bool(save_path), filename=save_path)
            
            # 获取数组信息
            import numpy as np
            np_array = np.array(result)
            
            response_text = f"成功解析分卷图片数据。\n"
            if save_path:
                response_text += f"解析后的数组已保存到: {save_path}\n"
            response_text += f"重组后数组形状: {np_array.shape}\n"
            response_text += f"数值范围: {np_array.min()} - {np_array.max()}\n"
            response_text += f"数组数据: {json.dumps(result, indent=2)}"
            
            return [types.TextContent(type="text", text=response_text)]
            
        elif name == "get_image_info":
            image_base64 = arguments["image_base64"]
            save_path = arguments.get("save_path")
            
            result = image_processor.get_image_info(image_base64, save=bool(save_path), filename=save_path)
            
            response_text = f"图片信息:\n"
            if save_path:
                response_text += f"信息已保存到: {save_path}\n"
            response_text += json.dumps(result, indent=2)
            
            return [types.TextContent(type="text", text=response_text)]
            
        elif name == "create_example_array":
            width = arguments.get("width", 100)
            height = arguments.get("height", 100)
            pattern = arguments.get("pattern", "gradient")
            channels = arguments.get("channels", 3)
            save_path = arguments.get("save_path")
            
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
            
            # 如果需要保存
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'data': result,
                        'shape': list(array.shape),
                        'pattern': pattern,
                        'channels': channels
                    }, f, indent=2, ensure_ascii=False)
            
            response_text = f"成功创建示例数组。\n"
            if save_path:
                response_text += f"数组已保存到: {save_path}\n"
            response_text += f"尺寸: {width}x{height}\n"
            response_text += f"通道数: {channels}\n"
            response_text += f"图案: {pattern}\n"
            response_text += f"数组形状: {array.shape}\n"
            response_text += f"数组数据: {json.dumps(result, indent=2)}"
            
            return [types.TextContent(type="text", text=response_text)]
            
        elif name == "file_to_array":
            file_path = arguments["file_path"]
            save_path = arguments.get("save_path")
            
            # 处理路径
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)
            
            result = image_processor.file_to_array(file_path, save=bool(save_path), filename=save_path)
            
            # 获取数组信息
            import numpy as np
            np_array = np.array(result)
            
            response_text = f"成功从文件读取并转换为数组。\n"
            response_text += f"文件路径: {file_path}\n"
            if save_path:
                response_text += f"数组已保存到: {save_path}\n"
            response_text += f"数组形状: {np_array.shape}\n"
            response_text += f"数值范围: {np_array.min()} - {np_array.max()}\n"
            response_text += f"数组数据: {json.dumps(result, indent=2)}"
            
            return [types.TextContent(type="text", text=response_text)]
            
        elif name == "save_base64_to_file":
            image_base64 = arguments["image_base64"]
            file_path = arguments["file_path"]
            format = arguments.get("format")
            
            # 保存文件
            saved_path = image_processor.save_base64_to_file(image_base64, file_path, format)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"成功保存图片文件。\n"
                         f"保存路径: {saved_path}"
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