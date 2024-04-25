
import pandas as pd
from config import DATA_DIR, DEBUG

if DEBUG: 
    import ipywidgets as widgets
    from IPython.display import display, HTML



html = str


def get_records(
    df, 
    tag, 
    text_len_gt=None,
    text_contains=None,
    text_not_contains=None,
    classname=None,
    classname_none=False, 
    classcount = None, 
    classcount_gt = None,
    classcount_lt = None,
    alt_len_gt=None,
    alt_len_lt=None,
    p0_id=None,
    p0_class=None,
    p0_tag=None,
    p1_id=None,
    p1_class=None,
    p1_tag=None,
    p2_id=None,
    p2_class=None,
    p2_tag=None,
    children_count_gt=None,
    children_count_lt=None,
    children_count_eq=None,
    children_h3_count_gt=None,
    children_h3_count_lt=None,
    children_h3_count_eq=None,
    children_h2_count_gt=None,
    children_h2_count_lt=None,
    children_h2_count_eq=None,
    ):
    idx = (df['tag'] == tag)
    if alt_len_gt is not None:
        idx = idx & (df['alt_len'] > alt_len_gt)
    if alt_len_lt is not None:
        idx = idx & (df['alt_len'] < alt_len_lt)
    if text_len_gt is not None:
        idx = idx & (df['text_len'] > text_len_gt)
    if text_contains is not None:
        idx = idx & (df['text'].str.contains(text_contains))
    if classname is not None:
        idx = idx & (df['class'].str.contains(classname))
    if classname_none == True:
        idx = idx & (df['class'].isna())
    if classcount is not None:
        idx = idx & (df['class_count'] == classcount)
    if classcount_gt is not None:
        idx = idx & (df['class_count'] > classcount_gt)
    if classcount_lt is not None:
        idx = idx & (df['class_count'] < classcount_lt)
    if p0_class is not None:
        idx = idx & (df['g0_parent_class'].str.contains(p0_class))
    if p0_id is not None:
        idx = idx & (df['g0_parent_id'].str.contains(p0_id))
    if p0_tag is not None:
        idx = idx & (df['g0_parent_tag'] == p0_tag)
    if p1_class is not None:
        idx = idx & (df['g1_parent_class'].str.contains(p1_class))
    if p1_id is not None:
        idx = idx & (df['g1_parent_id'].str.contains(p1_id))
    if p1_tag is not None:
        idx = idx & (df['g1_parent_tag'] == p1_tag)
    if p2_class is not None:
        idx = idx & (df['g2_parent_class'].str.contains(p2_class))
    if p2_id is not None:
        idx = idx & (df['g2_parent_id'].str.contains(p2_id))
    if p2_tag is not None:
        idx = idx & (df['g2_parent_tag'] == p2_tag)
    if children_count_gt is not None:
        idx = idx & (df['children_count'] > children_count_gt)
    if children_count_lt is not None:
        idx = idx & (df['children_count'] < children_count_lt)
    if children_count_eq is not None:
        idx = idx & (df['children_count'] == children_count_eq)
    if children_h3_count_gt is not None:
        idx = idx & (df['children_h3_count'] > children_h3_count_gt)
    if children_h3_count_lt is not None:
        idx = idx & (df['children_h3_count'] < children_h3_count_lt)
    if children_h3_count_eq is not None:
        idx = idx & (df['children_h3_count'] == children_h3_count_eq)
    if children_h2_count_gt is not None:
        idx = idx & (df['children_h2_count'] > children_h2_count_gt)
    if children_h2_count_lt is not None:
        idx = idx & (df['children_h2_count'] < children_h2_count_lt)
    if children_h2_count_eq is not None:
        idx = idx & (df['children_h2_count'] == children_h2_count_eq)

    return df[idx]
    
def h1_split(df): 
    h1_df = get_records(df, 'h1')
    df_list = []
    idx = list(h1_df.index) + [list(df.index)[-1] + 1]
    for i in range(len(idx) - 1):
        sub_df = df[idx[i]:idx[i+1]].copy()
        sub_df['article_title'] = df.iloc[idx[i]]['text']
        sub_df = sub_df.reset_index(drop=True)
        df_list.append(sub_df)
    return df_list
        
def auto_label(
    df,
    is_h1_split=True,
    img_class=None,
    img_p0_class=None,
    title_class=None,
    title_class_none=None,
    title_p0_class=None,
    title_p0_id=None,
    title_p0_tag=None,
    title_p1_class=None,
    title_p1_id=None,
    title_p1_tag=None,
    title_p2_class=None,
    title_p2_id=None,
    title_p2_tag=None,
    title_children_count_gt=None,
    title_children_count_lt=None,
    title_children_count_eq=None,
    title_children_h3_count_gt=None,
    title_children_h3_count_lt=None,
    title_children_h3_count_eq=None,
    title_children_h2_count_gt=None,
    title_children_h2_count_lt=None,
    title_children_h2_count_eq=None,
    p_class=None,
    p_class_none=None,
    p_text_len_gt=None, 
    p_p0_id=None,
    p_p0_class=None,
    p_p0_tag=None,
    p_p1_id=None,
    p_p1_class=None,
    p_p1_tag=None,
    p_p2_id=None,
    p_p2_class=None,
    p_p2_tag=None,
    p_children_count_gt=None,
    p_children_count_lt=None,
    p_children_count_eq=None,
    p_children_h3_count_gt=None,
    p_children_h3_count_lt=None,
    p_children_h3_count_eq=None,
    p_children_h2_count_gt=None,
    p_children_h2_count_lt=None,
    p_children_h2_count_eq=None,    
    p_tag='p', 
    title_tag='h2', 
    article_title_tag='h1',
    article_title_text_contains=None,
    article_title_class=None,
    unknown_text_list=[]
    ):
    
    article_df = h1_split(df) if is_h1_split else [df]
    dfs = []
    for i, sub_df in enumerate(article_df):
        sub_df['label'] = 'unknown'
        at_df = get_records(article_df[i], article_title_tag, classname=article_title_class, text_contains=article_title_text_contains) 
        sub_df.loc[at_df.index, 'label'] = 'article_title'        
        #!image
        img_df = get_records(
            article_df[i], 
            'img', 
            classname=img_class,
            p0_class=img_p0_class
        )
        sub_df.loc[img_df.index, 'label'] = 'image'
        #!paragraph
        p_df = get_records(
            article_df[i], 
            p_tag, 
            classname=p_class, 
            classname_none= p_class_none,
            text_len_gt=p_text_len_gt,
            p0_id=p_p0_id, 
            p0_class=p_p0_class, 
            p0_tag=p_p0_tag,
            p1_id=p_p1_id,
            p1_class=p_p1_class,
            p1_tag=p_p1_tag,
            p2_id=p_p2_id,
            p2_class=p_p2_class,
            p2_tag=p_p2_tag,
            children_count_gt=p_children_count_gt,
            children_count_lt=p_children_count_lt,
            children_count_eq=p_children_count_eq,
            children_h3_count_gt=p_children_h3_count_gt,
            children_h3_count_lt=p_children_h3_count_lt,
            children_h3_count_eq=p_children_h3_count_eq,
            children_h2_count_gt=p_children_h2_count_gt,
            children_h2_count_lt=p_children_h2_count_lt,
            children_h2_count_eq=p_children_h2_count_eq
            
        )
        sub_df.loc[p_df.index, 'label'] = 'paragraph'
        #!title
        t_df = get_records(
            article_df[i], 
            title_tag, 
            classname=title_class,
            classname_none=title_class_none, 
            p0_class=title_p0_class, 
            p0_id=title_p0_id, 
            p0_tag=title_p0_tag,
            p1_class=title_p1_class,
            p1_id=title_p1_id,
            p1_tag=title_p1_tag,
            p2_class=title_p2_class,
            p2_id=title_p2_id,
            p2_tag=title_p2_tag,
            children_count_gt=title_children_count_gt,
            children_count_lt=title_children_count_lt,
            children_count_eq=title_children_count_eq,
            children_h3_count_gt=title_children_h3_count_gt,
            children_h3_count_lt=title_children_h3_count_lt,
            children_h3_count_eq=title_children_h3_count_eq,    
            children_h2_count_gt=title_children_h2_count_gt,
            children_h2_count_lt=title_children_h2_count_lt,
            children_h2_count_eq=title_children_h2_count_eq
        )
        sub_df.loc[t_df.index, 'label'] = 'title'
        dfs.append(sub_df)
        for ut in unknown_text_list:
            u_tag = ut[0]
            u_text = ut[1]
            u_label = ut[2] if len(ut) == 3 else 'unknown'
            sub_df.loc[(sub_df['tag'] == u_tag) & (sub_df['text'].str.contains(u_text)), 'label'] = u_label
        
    for df in dfs:
        print(df.iloc[0]['source'])
        print(df['label'].value_counts())
    return [dfs[0]]


def print_article(df, img=True, use_prediction=False):
    if 'source' in df.columns:
        print(df.iloc[0]['source'])
    column = 'prediction' if use_prediction else 'label'
    relevant_df = df[df[column] != 'unknown']
    for i, row in relevant_df.iterrows():
        # if 'commended' in row['text']:
        #     print('---------------------------') 
        if row[column] == 'article_title':
            display(HTML(f"<h1>{row['text']}</h1>"))
        if row[column] == 'image' and img:
            display(HTML(f"<img src='{row['src']}'/>"))
        elif row[column] == 'paragraph':
            display(HTML(f"<p>{row['text']}</p>"))
        elif row[column] == 'title':
            display(HTML(f"<h2>{row['text']}</h2>"))




def page_div_html(p, l, color, is_current):    
    # bg_color = 'LightSkyBlue' if is_current else 'none'
    curr_page_style = 'background-color: LightSkyBlue; border-radius: 3px; padding: 1px' if is_current else 'background-color: white'
    component: html = f"""
        <span id="{p}_{l}" style="{curr_page_style}">
            (<span style="color: black">{p}<span>,
            <span style="color: {color}" >{l}<span>)
        </span>
    """
    return component

def color_unlabeled(unlabeled_count, curr_page, page, page_size):
    t = f'{curr_page},{unlabeled_count} '
    bc = 'on_blue' if curr_page == page else None
    color = 'red'
    if unlabeled_count == 0:
        color = 'green'
    elif unlabeled_count / page_size < 0.5:
        color = 'orange'
    return page_div_html(curr_page, unlabeled_count, color, curr_page == page)

def page_stats(df, page_size, page):
    label_status = ''
    for i in range(0, len(df), page_size):
        sub_df = df[i: i + page_size]
        unlabeled_count = len(sub_df[sub_df['label'] == 'not labeled'])
        unlabeled_ratio = unlabeled_count / page_size
        is_currant_page = i // page_size == page
        curr_page = i // page_size
        label_status += color_unlabeled(unlabeled_count, curr_page, page, page_size)
    label_status_html= f"""
        <div>
            <div>indexs: {page * page_size} - {page * page_size + page_size - 1} </div>
            {label_status}
        </div>
    """
    # print(label_status_html)
    return widgets.HTML(value=label_status_html)






def dataset_labeling(df, page=0, page_size=20, save_df=None):
    
    widget_list = []


    def handle_change(change):
        idx = page * page_size + widget_list.index(change.owner)
        df.loc[idx, 'label'] = change.new
        df.loc[idx, 'user_labeled'] = True
        save_df(df)
        

    def present_rows(from_idx=0, to_idx=50):
        row_widgets = []
        for idx, row in df[from_idx:to_idx].iterrows():
            selection = widgets.ToggleButtons(
                options=['image', 'title', 'paragraph', 'unknown'],
                description='label:',
                value=None if row['label'] == 'not labeled' else row['label'],
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                # tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
            #     icons=['check'] * 3
            )

            widget_list.append(selection)
            header_html: html = f"""
                <span style="background-color: gray; color: white; font-weight: 600">{idx})</span>
                prediction <span style="background-color: black; color: yellow">{row['prediction']}</span>
            """
            header = widgets.HTML(header_html)
            selection.observe(handle_change, names='value')
            raw_element = widgets.HTML(f"""<div style="width: 300px; height: auto; max-height: 300px;">
                {'&lt;'.join(row['elements'].split('<'))}
            </div>""")
            raw_accordion = accordion = widgets.Accordion(children=[raw_element], selected_index=None)
            accordion.set_title(0, 'raw')
            element_style = "border: solid; width: 300px; height: auto; max-height: 300px; img.resize { width: 300px; height: auto; }"
            element_html = widgets.HTML(f"""<div style="{element_style}">{row['elements']}</div>""")
            element_wg = widgets.HBox([element_html, raw_accordion])
            vbox = widgets.VBox([header, element_wg, selection])
            row_widgets.append(vbox)
            row_widgets.append(widgets.HTML('<hr>'))
        return widgets.VBox(row_widgets)
        # return row_widgets


    return present_rows(from_idx=page*page_size, to_idx=(page+1)*page_size)




def pageinated_slider(filename, page_size=20):
    df = pd.read_csv(DATA_DIR / 'scrapping' / 'labeled' / filename)
    def save_df(df):
        df.to_csv(DATA_DIR / 'scrapping' / 'labeled' / filename, index=False)
    def page_select(p):
        ps = page_stats(df, page_size, p)
        dl = dataset_labeling(df, page=p, page_size=page_size, save_df=save_df)
        return widgets.VBox([ps, dl])
        
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(df) / page_size,
        step=1,
    )

    widgets.interact(page_select, p=slider, continuous_update=False)

    