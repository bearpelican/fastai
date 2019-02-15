import inspect
from IPython.core.display import display, Markdown, HTML

from .nbtest import _submodule_name, get_tests_dir, separate_comp, get_qualname
from .nbdoc import fn_name, get_module_name
from ..imports.core import *
from .core import ifnone

__all__ = ['show_skeleton_test', 'show_md_code', 'create_skeleton_test', 'save_test', 'test_filename']

def show_skeleton_test(elt):
    "Display skeleton test in notebook"
    content = create_skeleton_test(elt)
    show_md_code(content)

def show_md_code(md):
    display(Markdown(f'```python\n{md}\n```'))

def create_skeleton_test(elt)->str:
    "Create markdown string for skeleton test"
    test_name = create_test_name(elt)
    test_def = f'def {test_name}():'
    content = create_skeleton_body(elt)
    return '\n\t'.join([test_def] + content)

def create_skeleton_body(elt)->List[str]:
    "Generates barebones test case for fastai `elt` function/class"
    spec = inspect.getfullargspec(elt)
    ann = spec.annotations
    lines = []
    args = [arg for arg in spec.args if arg not in ['self', 'cls']]
    required_args = get_required_args(elt)
    for arg,arg_type in required_args.items():
        if arg_type is None:
            lines.append(f'{arg} = # instantiate param \'{arg}\'')
        else:
            arg_type_name = fn_name(arg_type)
            lines.append(f'{arg}:{arg_type_name} = {arg_type_name}({stringify_required_args(arg_type)}) # instantiate param \'{arg}\'')

    params = ', '.join(required_args.keys())
    return_type = ann.get('return')
    return_type = f':{fn_name(return_type)}' if return_type else ''
    if is_instance_method(elt): # must create instance method of class first
        name_parts = elt.__qualname__.split('.')
        fn_class = inspect.getmodule(elt).__dict__[name_parts[0]]
        lines.append(f'instance = {name_parts[0]}({stringify_required_args(fn_class)})')
        lines.append(f'result = instance.{name_parts[-1]}({params})')
    else:
        lines.append(f'result{return_type} = {elt.__qualname__}({params})')
    lines.append(f'assert result == None, f"Failed: {elt.__name__} return unexpected result:'+' {result}"')
    return lines

def stringify_required_args(elt)->str:
    "Formats required args with arg:arg_type"
    required_args = get_required_args(elt)
    return ', '.join([k if v is None else f'{k}:{fn_name(v)}' for k,v in required_args.items()])

def get_required_args(elt)->Dict[str,TypeVar]:
    "Gathers only minimal args needed to run a given function."
    try: spec = inspect.getfullargspec(elt)
    except TypeError as e: return {}
    ann = spec.annotations
    required_args = spec.args
    if spec.defaults: required_args = required_args[:-len(spec.defaults)]
    required_args = [arg for arg in required_args if arg not in ['self', 'cls']]
    return {arg:ann.get(arg, None) for arg in required_args}

def is_instance_method(elt)->bool:
    "Returns False for class method and True for instance method"
    return len(elt.__qualname__.split('.')) > 1 and not inspect.ismethod(elt)

def create_test_name(elt)->str:
    "Generates default test function name for fastai `elt`"
    if inspect.isclass(elt): raise Exception('Generic class tests are not recommended. Please test class functions/methods individually')
    top,name = separate_comp(get_qualname(elt))
    test_name = '_'.join(top+[name])
    return f'test_{test_name}'

def skeleton_headers(elt)->str:
    "Returns module imports for new test file"
    module_name = get_module_name(elt)
    parts = module_name.split('.')
    path,name = '.'.join(parts[:-1]), parts[-1]
    contents = (f'import pytest\n'
                f'from {module_name} import *\n'
                f'from {path} import {name}\n')
    return contents

def save_test(elt, test_function):
    "Saves `test_function` in the `fastai/tests` directory for a given fastai `elt`"
    testfile = test_filename(elt)
    if testfile.exists():
        print('Test file already exists. Appending test function to end')
        with open(testfile, 'r') as f:
            file_contents = f.read()
    else:
        print('Could not find test file. Creating new one:', testfile)
        file_contents = skeleton_headers(elt)
    if test_function.__name__ in file_contents:
        print(f'Test collision: test already exists with {test_function.__name__}. Please rename or remove old one')
        return testfile
    source_code = inspect.getsource(test_function)
    file_contents = f'{file_contents}\n{source_code}'
    with open(testfile, 'w') as f:
        f.write(file_contents)
    return testfile

def test_filename(elt)->str:
    "Returns the file location where the tests of an api function should be located"
    subdir = _submodule_name(elt)
    subdir = f'{subdir}_' if subdir is not None else ''
    inspect.getmodule(elt)
    suffix = Path(inspect.getmodule(elt).__file__).name
    fn = get_tests_dir(elt)/f'test_{subdir}{suffix}'
    return fn