# import pytest
# from pipelines.text_process_graphs import load_csv_data, generate_content_map, get_data_dir_contents
# from config import OUT_DIR
# import glob
# import os

# def test_data_dir_contents():
#     files = get_data_dir_contents()
#     # assert len(files) > 1
#     # for f in files:
#     #     assert f[-3:] == 'xml'



# def test_generate_content_map():
#     filenames =[os.path.basename(f) for f in glob.glob(os.path.join(OUT_DIR,'*')) if f != 'content_map.json']    
#     f = generate_content_map(filenames)
#     assert len(f) > 0


# def test_generate_audio_content_map():
#     pass



# def test_load_csv_file_test():
#     # data = load_csv_data.execute_in_process(run_config = {
#     #     'ops': {
#     #         'load_csv_data': {
#     #             'config': {
#     #                 "filepath": "data/csv/wine_openers.csv"
#     #             }
#     #         }
#     #     }
#     # })
#     data = load_csv_data()
#     assert len(data) > 0


# def test_ai21_jobs():
#     pass