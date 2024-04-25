from glob import glob
import pytest
from bs4 import BeautifulSoup
import pandas as pd
from config import DATA_DIR
from components.scrapping.feature_utils import get_all_soups, get_feature_df, get_hpi_elements, get_attr_set, sibiling_data, get_g_parent_attr
from components.scrapping.labeling_utils import auto_label

html = str

TEST_URL = "https://test-url.com"
TEST_HTML: html = """
<html>
    <head>
        <title>Test</title>
    </head>
    <body>
        <div>
            <div>
                <h1 class="h1-class">title</h1>
                        <p class="p-class" >this is content paragraph1</p>
                    <h2 class="h2-class" alt="alt2 alt2">sub title 1</h2>
                        <p class="p-class1 p-class2" alt="alt1" >this is content paragraph2</p>
                    <h2 class="h2-class" >sub title 2</h2>
                        <p class="p-class p-class3 p-class3" >this is content paragraph3</p>
                            <img src="http://test-site.com/image1.jpg" alt="image1" />
                        <div class="div-class" >div paragraph</div>
            </div>
        </div>
    </body>
</html>
"""


def test_feature_prediction():
    soap = BeautifulSoup(TEST_HTML, 'html.parser')
    features = get_feature_df(soap, TEST_URL)
    print(features)
    h1 = features[features['tag'] == 'h1']
    h2 = features[features['tag'] == 'h2']
    p = features[features['tag'] == 'p']
    img = features[features['tag'] == 'img']
    assert len(h1) == 1
    assert len(h2) == 2
    assert len(p) == 3
    assert len(img) == 1
    h1.loc[0, 'text'] == 'title'
    assert h2.loc[0,'class']
    assert h1.loc[0,'class'] == 'h1-class'
    assert h1.loc[0,'class_count'] == 1
    assert h2.loc[0,'class_count'] == 1
    assert h2.loc[1,'class_count'] == 1
    assert h2.loc[0, 'text'] == 'sub title 1'
    assert h2.loc[1, 'text'] == 'sub title 2'
    assert h2.loc[0, 'class'] == 'h2-class'
    


def test_auto_labels():
    for idx, f in enumerate(glob(str(DATA_DIR / 'scrapping' / 'unlabeled' / '*.csv'))):
        if 'news.parenting101.com-en-makeup-internet-worst' in f:
            df = pd.read_csv(f)
            pds = auto_label(df, 
                img_class='gallery-img-s',
                title_class='gallery-item-title',
                p_class='gallery-item-description',
                p_tag='div',
                )
            assert len(pds) == 3
            vc = pds[0]['label'].value_counts()
            assert vc['title'] == 50
            assert vc['paragraph'] == 100
            assert vc['image'] == 50
            assert vc['article_title'] == 1
        elif 'boredpanda.com-funny-conversations-overheard-bumble' in f:
            df = pd.read_csv(f)
            #https://www.boredpanda.com/funny-conversations-overheard-bumble/
            pds = auto_label(df, 
                img_class='image-size-full', 
                p_p0_id='list-text', 
                )
            vc = pds[0]['label'].value_counts()  
            assert vc['paragraph'] == 16
            assert vc['image'] == 50
            assert vc['article_title'] == 1

        elif 'dallas-now-cast' in f:
            df = pd.read_csv(f)
            pds = auto_label(df, 
                img_class='gallery-item-img', 
                p_class='gallery-item-description', 
                title_class='gallery-item-title',
                p_tag='div',
            )
            vc = pds[0]['label'].value_counts()
            assert vc['title'] == 40
            assert vc['paragraph'] == 80
            assert vc['image'] == 40
            assert vc['article_title'] == 1