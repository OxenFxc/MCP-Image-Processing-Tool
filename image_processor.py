import numpy as np
from PIL import Image
import base64
import io
import json
from typing import List, Dict, Any, Tuple, Optional
import math
import os

def mcp_array_to_image(data: List[List[List[int]]], format: str = "PNG", save_path: str = None) -> str:
    """MCP工具：将3D数组转换为图片的base64编码，并可选择保存到文件"""
    processor = ImageProcessor()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        return processor.array_to_image(data, format=format, save=True, filename=save_path)
    return processor.array_to_image(data, format=format, save=False)

def mcp_image_to_array(image_base64: str, save_path: str = None) -> List[List[List[int]]]:
    """MCP工具：将base64编码的图片转换为3D数组，并可选择保存为JSON"""
    processor = ImageProcessor()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        return processor.image_to_array(image_base64, save=True, filename=save_path)
    return processor.image_to_array(image_base64, save=False)

def mcp_create_chunked_image(data: List[List[List[int]]], format: str = "PNG", save_path: str = None) -> Dict[str, Any]:
    """MCP工具：创建分卷图片数据，并可选择保存到文件"""
    processor = ImageProcessor()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        return processor.create_chunked_image(data, format=format, save=True, filename=save_path)
    return processor.create_chunked_image(data, format=format, save=False)

def mcp_parse_chunked_image(chunked_data: Dict[str, Any], save_path: str = None) -> List[List[List[int]]]:
    """MCP工具：解析分卷图片数据，并可选择保存结果"""
    processor = ImageProcessor()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        return processor.parse_chunked_image(chunked_data, save=True, filename=save_path)
    return processor.parse_chunked_image(chunked_data, save=False)

def mcp_get_image_info(image_base64: str, save_path: str = None) -> Dict[str, Any]:
    """MCP工具：获取图片信息，并可选择保存到文件"""
    processor = ImageProcessor()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        return processor.get_image_info(image_base64, save=True, filename=save_path)
    return processor.get_image_info(image_base64, save=False)

def mcp_file_to_array(file_path: str, save_path: str = None) -> List[List[List[int]]]:
    """MCP工具：从文件路径读取图片并转换为3D数组，可选择保存结果"""
    processor = ImageProcessor()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        return processor.file_to_array(file_path, save=True, filename=save_path)
    return processor.file_to_array(file_path, save=False)

def mcp_save_base64_to_file(image_base64: str, file_path: str, format: str = None) -> str:
    """MCP工具：将base64编码的图片保存为文件"""
    processor = ImageProcessor()
    return processor.save_base64_to_file(image_base64, file_path, format)

class ImageProcessor:
    """处理图片和数组之间的转换，支持分卷处理"""
    
    def __init__(self, chunk_size: int = 1024 * 1024, output_dir: str = "output"):  # 1MB 默认分块大小
        self.chunk_size = chunk_size
        self.max_dimension = 2048  # 最大图片尺寸限制
        self.output_dir = output_dir
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def array_to_image(self, data: List[List[List[int]]], format: str = "PNG", save: bool = True, filename: str = None) -> str:
        """
        将3D数组转换为图片的base64编码，并可选择保存到文件
        
        Args:
            data: 3D数组 [height, width, channels]
            format: 图片格式 (PNG, JPEG等)
            save: 是否保存到文件
            filename: 保存的文件名（如果为None则自动生成）
            
        Returns:
            base64编码的图片字符串
        """
        try:
            # 转换为numpy数组
            np_array = np.array(data, dtype=np.uint8)
            
            # 检查数组维度
            if len(np_array.shape) == 2:
                mode = 'L'
            elif len(np_array.shape) == 3:
                if np_array.shape[2] == 1:
                    np_array = np_array.squeeze(axis=2)
                    mode = 'L'
                elif np_array.shape[2] == 3:
                    mode = 'RGB'
                elif np_array.shape[2] == 4:
                    mode = 'RGBA'
                else:
                    raise ValueError(f"不支持的通道数: {np_array.shape[2]}")
            else:
                raise ValueError(f"不支持的数组维度: {np_array.shape}")
            
            # 创建PIL图片
            image = Image.fromarray(np_array, mode=mode)
            
            # 转换为base64
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # 如果需要保存到文件
            if save:
                if filename is None:
                    filename = f"array_to_image_{len(np_array)}x{len(np_array[0])}_{mode}.{format.lower()}"
                file_path = os.path.join(self.output_dir, filename)
                image.save(file_path, format=format)
                print(f"图片已保存到: {file_path}")
            
            return image_base64
            
        except Exception as e:
            raise Exception(f"数组转图片失败: {str(e)}")
    
    def image_to_array(self, image_base64: str, save: bool = True, filename: str = None) -> List[List[List[int]]]:
        """
        将base64编码的图片转换为3D数组，并可选择保存为JSON或NPY文件
        
        Args:
            image_base64: base64编码的图片字符串
            save: 是否保存到文件
            filename: 保存的文件名（如果为None则自动生成）
            
        Returns:
            3D数组 [height, width, channels]
        """
        try:
            # 解码base64
            image_data = base64.b64decode(image_base64)
            
            # 创建PIL图片
            image = Image.open(io.BytesIO(image_data))
            
            # 确保图片模式正确
            if image.mode not in ['L', 'RGB', 'RGBA']:
                image = image.convert('RGB')
            
            # 转换为numpy数组并确保类型为uint8
            np_array = np.array(image, dtype=np.uint8)
            
            # 确保是3D数组
            if len(np_array.shape) == 2:
                np_array = np.expand_dims(np_array, axis=2)
            
            # 验证数值范围
            if np_array.min() < 0 or np_array.max() > 255:
                raise ValueError("图片数组值必须在0-255范围内")
            
            # 转换为Python列表
            array_data = np_array.tolist()
            
            # 如果需要保存到文件
            if save:
                if filename is None:
                    filename = f"image_to_array_{image.width}x{image.height}.json"
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'data': array_data,
                        'shape': list(np_array.shape),
                        'mode': image.mode,
                        'format': image.format
                    }, f, indent=2, ensure_ascii=False)
                print(f"数组已保存到: {file_path}")
            
            return array_data
            
        except Exception as e:
            raise Exception(f"图片转数组失败: {str(e)}")
    
    def create_chunked_image(self, data: List[List[List[int]]], format: str = "PNG", save: bool = True, filename: str = None) -> Dict[str, Any]:
        """
        创建分卷图片数据，并可选择保存到文件
        
        Args:
            data: 大图片的3D数组
            format: 图片格式
            save: 是否保存到文件
            filename: 保存的文件名（如果为None则自动生成）
            
        Returns:
            包含分卷信息的字典
        """
        try:
            np_array = np.array(data, dtype=np.uint8)
            height, width = np_array.shape[:2]
            
            # 检查是否需要分卷
            total_pixels = height * width
            if len(np_array.shape) == 3:
                total_pixels *= np_array.shape[2]
            
            result = {}
            if total_pixels * 4 <= self.chunk_size:  # 不需要分卷
                image_base64 = self.array_to_image(data, format, save=False)
                result = {
                    "is_chunked": False,
                    "data": image_base64,
                    "original_shape": [height, width] + ([] if len(np_array.shape) == 2 else [np_array.shape[2]]),
                    "format": format
                }
            else:
                # 计算分块大小
                chunk_height = min(height, int(math.sqrt(self.chunk_size / (width * (np_array.shape[2] if len(np_array.shape) == 3 else 1)))))
                chunk_height = max(1, chunk_height)
                
                chunks = []
                chunk_info = []
                
                for start_row in range(0, height, chunk_height):
                    end_row = min(start_row + chunk_height, height)
                    chunk_data = np_array[start_row:end_row]
                    
                    # 转换为base64
                    chunk_base64 = self.array_to_image(chunk_data.tolist(), format, save=False)
                    chunks.append(chunk_base64)
                    
                    chunk_info.append({
                        "start_row": start_row,
                        "end_row": end_row,
                        "shape": list(chunk_data.shape)
                    })
                
                result = {
                    "is_chunked": True,
                    "chunks": chunks,
                    "chunk_info": chunk_info,
                    "original_shape": list(np_array.shape),
                    "total_chunks": len(chunks),
                    "format": format
                }
            
            # 如果需要保存到文件
            if save:
                if filename is None:
                    filename = f"chunked_image_{height}x{width}.json"
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"分卷数据已保存到: {file_path}")
            
            return result
            
        except Exception as e:
            raise Exception(f"创建分卷图片失败: {str(e)}")
    
    def parse_chunked_image(self, chunked_data: Dict[str, Any], save: bool = True, filename: str = None) -> List[List[List[int]]]:
        """
        解析分卷图片数据，并可选择保存结果
        
        Args:
            chunked_data: 分卷数据字典
            save: 是否保存到文件
            filename: 保存的文件名（如果为None则自动生成）
            
        Returns:
            完整的3D数组
        """
        try:
            if not chunked_data.get("is_chunked", False):
                array_data = self.image_to_array(chunked_data["data"])
            else:
                original_shape = chunked_data["original_shape"]
                chunks = chunked_data["chunks"]
                chunk_info = chunked_data["chunk_info"]
                
                # 创建结果数组
                result = np.zeros(original_shape, dtype=np.uint8)
                
                # 组合分块
                for chunk_base64, info in zip(chunks, chunk_info):
                    chunk_array = np.array(self.image_to_array(chunk_base64, save=False), dtype=np.uint8)
                    
                    if len(original_shape) == 3 and len(chunk_array.shape) == 3 and chunk_array.shape[2] == 1:
                        chunk_array = chunk_array.squeeze(axis=2)
                    elif len(original_shape) == 2 and len(chunk_array.shape) == 3:
                        chunk_array = chunk_array.squeeze(axis=2)
                    
                    start_row = info["start_row"]
                    end_row = info["end_row"]
                    result[start_row:end_row] = chunk_array
                
                array_data = result.tolist()
            
            # 如果需要保存到文件
            if save:
                if filename is None:
                    shape_str = 'x'.join(map(str, chunked_data.get("original_shape", [])))
                    filename = f"parsed_chunked_image_{shape_str}.json"
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'data': array_data,
                        'shape': chunked_data.get("original_shape", []),
                        'format': chunked_data.get("format", "PNG")
                    }, f, indent=2, ensure_ascii=False)
                print(f"解析后的数组已保存到: {file_path}")
            
            return array_data
            
        except Exception as e:
            raise Exception(f"解析分卷图片失败: {str(e)}")
    
    def get_image_info(self, image_base64: str, save: bool = True, filename: str = None) -> Dict[str, Any]:
        """
        获取图片信息，并可选择保存到文件
        
        Args:
            image_base64: base64编码的图片
            save: 是否保存到文件
            filename: 保存的文件名（如果为None则自动生成）
            
        Returns:
            图片信息字典
        """
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            info = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "size_bytes": len(image_data)
            }
            
            # 如果需要保存到文件
            if save:
                if filename is None:
                    filename = f"image_info_{image.width}x{image.height}.json"
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
                print(f"图片信息已保存到: {file_path}")
            
            return info
            
        except Exception as e:
            raise Exception(f"获取图片信息失败: {str(e)}")
    
    def file_to_array(self, file_path: str, save: bool = True, filename: str = None) -> List[List[List[int]]]:
        """
        从文件路径读取图片并转换为3D数组，可选择保存结果
        
        Args:
            file_path: 图片文件路径
            save: 是否保存到文件
            filename: 保存的文件名（如果为None则自动生成）
            
        Returns:
            3D数组 [height, width, channels]
        """
        try:
            # 读取图片文件
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # 转换为base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 使用现有方法转换为数组并保存
            return self.image_to_array(image_base64, save=save, filename=filename)
            
        except Exception as e:
            raise Exception(f"文件转数组失败: {str(e)}")
    
    def save_base64_to_file(self, image_base64: str, file_path: str, format: str = None) -> str:
        """
        将base64编码的图片保存为文件
        
        Args:
            image_base64: base64编码的图片字符串
            file_path: 保存的文件路径
            format: 图片格式（可选，如果不指定则从文件扩展名推断）
            
        Returns:
            保存的文件的绝对路径
        """
        try:
            # 解码base64
            image_data = base64.b64decode(image_base64)
            
            # 创建PIL图片对象
            image = Image.open(io.BytesIO(image_data))
            
            # 如果没有指定格式，从文件扩展名推断
            if format is None:
                format = os.path.splitext(file_path)[1].lstrip('.')
                if not format:
                    format = 'PNG'  # 默认使用PNG
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 保存图片
            image.save(file_path, format=format.upper())
            print(f"图片已保存到: {file_path}")
            
            return os.path.abspath(file_path)
            
        except Exception as e:
            raise Exception(f"保存图片文件失败: {str(e)}")
    
    def save_array_to_file(self, array_data: List[List[List[int]]], file_path: str, format: str = "PNG") -> str:
        """
        将3D数组保存为图片文件
        
        Args:
            array_data: 3D数组 [height, width, channels]
            file_path: 保存的文件路径
            format: 图片格式（默认PNG）
            
        Returns:
            保存的文件的绝对路径
        """
        try:
            # 转换为numpy数组并确保类型为uint8
            np_array = np.array(array_data, dtype=np.uint8)
            
            # 检查数组维度
            if len(np_array.shape) == 2:
                mode = 'L'
            elif len(np_array.shape) == 3:
                if np_array.shape[2] == 1:
                    np_array = np_array.squeeze(axis=2)
                    mode = 'L'
                elif np_array.shape[2] == 3:
                    mode = 'RGB'
                elif np_array.shape[2] == 4:
                    mode = 'RGBA'
                else:
                    raise ValueError(f"不支持的通道数: {np_array.shape[2]}")
            else:
                raise ValueError(f"不支持的数组维度: {np_array.shape}")
            
            # 验证数值范围
            if np_array.min() < 0 or np_array.max() > 255:
                raise ValueError("图片数组值必须在0-255范围内")
            
            # 创建PIL图片
            image = Image.fromarray(np_array, mode=mode)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 保存图片
            image.save(file_path, format=format)
            
            return os.path.abspath(file_path)
            
        except Exception as e:
            raise Exception(f"保存数组到文件失败: {str(e)}")
    
    def save_array(self, array_data: List[List[List[int]]], file_path: str, format: str = "npy") -> str:
        """
        直接保存数组数据到文件
        
        Args:
            array_data: 3D数组 [height, width, channels]
            file_path: 保存的文件路径
            format: 保存格式，支持 "npy"（numpy格式）或 "json"
            
        Returns:
            保存的文件的绝对路径
        """
        try:
            # 转换为numpy数组
            np_array = np.array(array_data, dtype=np.uint8)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            if format.lower() == "npy":
                # 如果没有.npy后缀，添加它
                if not file_path.lower().endswith('.npy'):
                    file_path += '.npy'
                # 保存为numpy格式
                np.save(file_path, np_array)
            elif format.lower() == "json":
                # 如果没有.json后缀，添加它
                if not file_path.lower().endswith('.json'):
                    file_path += '.json'
                # 保存为JSON格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'shape': np_array.shape,
                        'data': np_array.tolist()
                    }, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的保存格式: {format}")
            
            return os.path.abspath(file_path)
            
        except Exception as e:
            raise Exception(f"保存数组数据失败: {str(e)}")
    
    def load_array(self, file_path: str) -> List[List[List[int]]]:
        """
        从文件加载数组数据
        
        Args:
            file_path: 数组数据文件路径
            
        Returns:
            3D数组 [height, width, channels]
        """
        try:
            if file_path.lower().endswith('.npy'):
                # 从numpy格式加载
                np_array = np.load(file_path)
            elif file_path.lower().endswith('.json'):
                # 从JSON格式加载
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    np_array = np.array(data['data'], dtype=np.uint8)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            # 验证数组
            if not (2 <= len(np_array.shape) <= 3):
                raise ValueError(f"不支持的数组维度: {np_array.shape}")
            
            # 确保是3D数组
            if len(np_array.shape) == 2:
                np_array = np.expand_dims(np_array, axis=2)
            
            return np_array.tolist()
            
        except Exception as e:
            raise Exception(f"加载数组数据失败: {str(e)}") 