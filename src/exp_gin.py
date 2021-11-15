import gin
import os


@gin.configurable
def func(c1):
    print(f"c1 = {c1}")
    return c1

if __name__ == '__main__':
    file_path = f'src/configs/materials.gin'
    gin.parse_config_file(file_path)
    func()
