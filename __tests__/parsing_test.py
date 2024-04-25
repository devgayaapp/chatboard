import pytest
from components.nerative.csv_nerative_builder import csv_nerative_builder
from components.nerative.nerative_builider import nerative_builder
from components.parsers.csv_parser import parse_csv_file

from components.wordpress import WordPressFile


scale = 3
screen_width = 1000
screen_height = int(( screen_width / 9 ) * 16)

screen_size = (screen_width, screen_height)

fps = 30
cap_size = (screen_height, screen_width)


def test_image_in_title():
    filepath = 'data/content-d70dc8ea-1827-4483-8fcf-9d39522af661.xml'
    title = 'Historical Notes And Mementos People Randomly Found'
    wordpress_file = WordPressFile(filepath)
    article = wordpress_file.search(title)
    screens = nerative_builder(article)
    pass



def test_image_before_text():
    #TODO the last element is not working here
    filepath = 'data/content-d70dc8ea-1827-4483-8fcf-9d39522af661.xml'
    title = 'The most memorable royal wedding scandals and weird moments'
    wordpress_file = WordPressFile(filepath)
    article = wordpress_file.search(title)
    screens = nerative_builder(article)
    pass



def test_for_non_complete_nerative():
    filepath = 'data/content-7a14c1e4-5d61-4611-ab40-e32f2104edbf.xml'
    title = '20 Richest People In UK'
    wordpress_file = WordPressFile(filepath)
    article = wordpress_file.search(title)
    screens = nerative_builder(article)
    print(len(screens))

def test_stress_test_for_full_file():
    files = [
        'data/content-d70dc8ea-1827-4483-8fcf-9d39522af661.xml',
        'data/content-7a14c1e4-5d61-4611-ab40-e32f2104edbf.xml'
    ]
    for filepath in files:
        wordpress_file = WordPressFile(filepath)
        for article in wordpress_file:
            screens = nerative_builder(article)



def test_csv_parsing_to_screens():
    filepath = "data/csv/wine_openers.csv"
    f = parse_csv_file(filepath)
    screens = csv_nerative_builder(f)
    assert len(screens) > 0
    assert screens[0].title != None
    assert screens[0].text == None
    assert screens[1].title == None
    assert screens[1].text != None
