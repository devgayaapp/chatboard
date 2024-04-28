# import pytest
# from components.adapters.google_adapter import GoogleAdapter
# import os
# from pathlib import Path
# from tempfile import TemporaryDirectory


# def test_google_adapter():
#     url = 'https://drive.google.com/file/d/1-DQM2R39RUbZfpOUZ4BH4Xqk1QPEb04T/view?usp=sharing'
#     adapter = GoogleAdapter()    
#     with TemporaryDirectory() as tmp_dir:
#         print(tmp_dir)
#         filepath, metadata = adapter.download_file(url, Path(tmp_dir))
#         assert filepath.exists()
    
    
# def test_google_adapter_list_files():
#     folder_url = 'https://drive.google.com/drive/u/1/folders/1NCzoFlNgdEmIo2y2bvQZhCtPDNDgMcCz'
#     adapter = GoogleAdapter()
#     ids = adapter.get_file_ids_from_folder(folder_url)
#     assert len(ids) == 7
    