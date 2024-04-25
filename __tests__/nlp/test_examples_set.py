import pytest

from components.prompt.examples import RedisExamples



INDEX = 'test_index'


# @pytest.fixture(autouse=True)
# def before_all_tests():
#     example_set = RedisExamples(INDEX)
#     example_set.index()


# @pytest.fixture(autouse=True)
# def after_all_tests():
#     example_set = RedisExamples(INDEX)
#     example_set.clear()
#     example_set.drop_index()

@pytest.fixture
def before_each_test():
    example_set = RedisExamples(INDEX)
    example_set.index()

@pytest.fixture
def after_each_test(request):
    def fin():
        example_set = RedisExamples(INDEX)
        example_set.clear()
        example_set.drop_index()
    request.addfinalizer(fin)


def test_basic_set_operations(before_each_test, after_each_test):
    example_set = RedisExamples(INDEX)
    example_set.add('hello world', 'this is a test')
    
    assert len(example_set) == 1

    example_set.add('goodbye world', 'this is another test')

    assert len(example_set) == 2

    all_examples = example_set.get_all()

    assert len(all_examples) == 2
    assert all_examples[0]['text'] == 'this is a test'
    assert all_examples[1]['text'] == 'this is another test'
    

    print(all_examples)

    keys = example_set.list_keys()
    ex1 = example_set.get(keys[0])
    assert ex1['text'] == 'this is a test'
    assert ex1['language'] == 'en'

    example_set.delete(keys[0])

    assert len(example_set) == 1

    example_set.add('hello world', 'this is a test', language='es')

    assert len(example_set) == 2

    examples = example_set.get_all()

    assert examples[0]['language'] == 'en'
    assert examples[1]['language'] == 'es'


def test_search_similar(before_each_test, after_each_test):
    example_set = RedisExamples(INDEX)
    example_set.add('cats are the best', 'cats are the best')
    example_set.add('I like cucambers', 'I like cucambers')
    example_set.add('the price of housing is rising', 'I like cucambers')

    sim_ex = example_set.similar('dogs are the best', num_results=1)

    print(sim_ex)
    assert sim_ex[0]['text'] == 'cats are the best'




def test_loading_example_file(before_each_test, after_each_test):
    example_set = RedisExamples(INDEX)

    example_set.populate_from_file('__tests__/nlp/test_data/examples_prompt.yaml')

    assert len(example_set) == 2

    examples = example_set.get_all()

    assert examples[0]['text'] == 'red, green, blue, yellow, orange\n'
    assert examples[1]['text'] == 'football, basketball, baseball, soccer, hockey\n'

    example_set.populate_from_file('__tests__/nlp/test_data/examples_prompt_update.yaml')

    assert len(example_set) == 3

    examples = example_set.get_all()

    assert examples[1]['text'] == 'dog, cat, horse, cow, pig\n'




def test_similar():

    RedisExamples.populate_from_file('__tests__/nlp/test_data/examples_prompt_update.yaml')

    example_set = RedisExamples.from_file('__tests__/nlp/test_data/examples_prompt_update.yaml')

    pets_examples = example_set.similar('pets', num_results=3)

    assert pets_examples[0]['text'] == 'dog, cat, horse, cow, pig\n'

    animals_examples = example_set.similar('animals', num_results=3)

    assert animals_examples[0]['text'] == 'dog, cat, horse, cow, pig\n'
    sports_examples = example_set.similar('sports', num_results=3)

    assert sports_examples[0]['text'] == 'football, basketball, baseball, soccer, hockey\n'

    

    print('sdfdsfas')


    RedisExamples.clear_from_file('__tests__/nlp/test_data/examples_prompt_update.yaml')

