import pytest
from components.prompt.examples import RedisExamples

from components.prompt.prompt_parser import CompletionError, complete_prompt, load_prompt, process_messages, prompt





def test_prompt():
    text = "Say Hello world"
    completion = complete_prompt(text)
    assert completion == "Hello world"


def test_prompt_yaml_parsing():
    text = "animals"
    prompt_config = load_prompt('__tests__/nlp/test_data/basic_prompt.yaml')
    assert len(prompt_config['system']) > 0
    msgs = process_messages(text, prompt_config)
    assert len(msgs) == 2
    

    prompt_config = load_prompt('__tests__/nlp/test_data/msgs_basic_prompt.yaml')
    msgs = process_messages(text, prompt_config)
    assert len(msgs) == 6
    assert msgs[1]['content'] == 'colors'
    assert msgs[1]['role'] == 'user'
    assert msgs[2]['content'] == 'red, green, blue, yellow, orange\n'
    assert msgs[2]['role'] == 'assistant'
    assert msgs[3]['content'] == 'sports'
    assert msgs[3]['role'] == 'user'


def test_prompt_yaml_completion():
    text = "animals"
    completion = complete_prompt(text, filepath='__tests__/nlp/test_data/basic_prompt.yaml')
    items = completion.split(',')
    print(completion)
    assert len(items) == 5


def test_prompt_params_yaml_completion():
    topic = "animals"
    min_length = 10
    max_length = 20
    text = "sports"
    params = {
        'topic': topic,
        'min_length': min_length,
        'max_length': max_length,
        'text': text,
    }
    prompt_config = load_prompt(filepath='__tests__/nlp/test_data/basic_prompt_params.yaml')
    msgs = process_messages(prompt_text='write me', prompt_config=prompt_config, params=params)

    assert len(msgs) == 2
    assert topic in msgs[0]['content']
    assert str(min_length) in msgs[0]['content']
    assert str(max_length) in msgs[0]['content']
    assert msgs[1]['content'] == 'write me'

    prompt_config = load_prompt(filepath='__tests__/nlp/test_data/basic_prompt_params_user.yaml')

    msgs = process_messages(prompt_config=prompt_config, params=params)

    assert len(msgs) == 2
    assert topic in msgs[0]['content']
    assert str(min_length) in msgs[0]['content']
    assert str(max_length) in msgs[0]['content']
    assert text in msgs[1]['content']

    msgs = process_messages(prompt_text='write me', prompt_config=prompt_config, params=params)

    assert len(msgs) == 2
    assert topic in msgs[0]['content']
    assert str(min_length) in msgs[0]['content']
    assert str(max_length) in msgs[0]['content']
    assert text in msgs[1]['content']
    assert 'write me' in msgs[1]['content']


def test_decorator_yaml_completion():

    @prompt(filename='basic_prompt.yaml')
    def generate_items(text, completion):
        items = completion.split(',')        
        return items, text
    
    text = "animals"
    items, comp_text = generate_items(text)
    print(items)
    assert len(items) == 5
    assert comp_text == text


@pytest.fixture
def before_example_test():
    RedisExamples.populate_from_file('__tests__/nlp/test_data/examples_prompt_update.yaml')

@pytest.fixture
def after_example_test(request):
    def fin():
        RedisExamples.clear_from_file('__tests__/nlp/test_data/examples_prompt_update.yaml')
    request.addfinalizer(fin)



def test_examples_yaml_completion(before_example_test, after_example_test):
    
    # completion = complete_prompt('pets', filepath='__tests__/nlp/test_data/examples_prompt_update.yaml', num_examples=1)
    completion = complete_prompt('pets', filepath='__tests__/nlp/test_data/examples_prompt_update.yaml', num_examples=1)

    assert 'ERROR' not in completion
    assert len(completion.split(',')) == 5
    


def test_error_yaml_completion(before_example_test, after_example_test):
    
    with pytest.raises(CompletionError) as excinfo:
        completion = complete_prompt('write me an article about cats', filepath='__tests__/nlp/test_data/examples_prompt_update.yaml', num_examples=1)
    assert str(excinfo.value) != '' and str(excinfo.value) is not None
    




