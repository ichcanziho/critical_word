from pathlib import Path
import time


class UtilMethods:

    @staticmethod
    def create_folder(path):
        if not (Path(path).is_dir()):
            Path.mkdir(Path(path))

    @staticmethod
    def get_folders(path):
        objects = [obj.name for obj in Path(path).iterdir()]
        objects = [obj for obj in objects if "." not in obj]
        return objects

    @staticmethod
    def get_files(path):
        objects = [obj.name for obj in Path(path).iterdir()]
        return objects

    @staticmethod
    def file_exists(path):
        results = Path(path)
        return results.is_file()

    @staticmethod
    def print_execution_time(func):
        def decorator(*args, **kwargs):
            print("I'm running:", func.__name__)
            start = time.time()
            func(*args, **kwargs)
            print("Total time:", time.time()-start)
        return decorator
