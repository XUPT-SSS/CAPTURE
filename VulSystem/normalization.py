import pandas as pd
import re
# from clean_gadget import clean_gadget
import clean_gadget
from preprocess.lang_processors.cpp_processor import CppProcessor


def normalization(source):
    """
    normalization code
    :param source: dataframe
    :return:
    """
    cpp_processor = CppProcessor()
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            line = re.sub('^#define.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget.clean_gadget([code])
        code[0] = re.sub('"".*?""', '', code[0], 20)
        code_list = cpp_processor.tokenize_code(code[0])
        print(len(code_list))

        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)
        # print(tokenization_code)
    return nor_code


def normalization2(source):
    cpp_processor = CppProcessor()
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        code[0] = re.sub('"".*?""', '', code[0], 20)

        code_list = cpp_processor.tokenize_code(code[0])
        # nor_code.append(code[0])
        # nor_code.append(code6)
        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)
        print(tokenization_code)
        with open('./corpus.txt', 'a') as f:
            f.write(tokenization_code)
            f.write('\n')
    return nor_code


def mutrvd():
    train = pd.read_pickle('trvd_train.pkl')
    test = pd.read_pickle('trvd_test.pkl')
    val = pd.read_pickle('trvd_val.pkl')

    train['code'] = normalization(train)
    train.to_pickle('./mutrvd/train.pkl')

    test['code'] = normalization(test)
    test.to_pickle('./mutrvd/test.pkl')

    val['code'] = normalization(val)
    val.to_pickle('./mutrvd/val.pkl')


def nor(source):
    cpp_processor = CppProcessor()
    nor_code = []

    #2023.6.7 Comment out the following line
    # source = re.sub(r'(?s)#.*?#endif', '', source)
    lines = source.split('\n')
    # print(lines)
    code = ''
    for line in lines:
        line = line.strip()
        line = re.sub('//.*', '', line)
        # 2023.6.7 Comment out the following line
        # line = re.sub(r'^#define.*', '', line)
        code += line + ' '
    # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
    code = re.sub('/\*.*?\*/', '', code)
    # code = re.sub('#.*?endif', '', code)
    code = clean_gadget.clean_gadget([code])

    # code[0] = code[0].replace('"".*?""', '', 10)
    # code[0] = re.sub('"".*?""', '', code[0], 20)
    # code[0] = re.sub('"".*""', '', code[0], 20)
    code_list = cpp_processor.tokenize_code(code[0])
    tokenization_code = ''
    for token in code_list:
        tokenization_code = tokenization_code + token + " "
    nor_code.append(tokenization_code)
    # print(tokenization_code)
    return nor_code


if __name__ == '__main__':
    str = r"""
static int qdm2_get_vlc (GetBitContext *gb, VLC *vlc, int flag, int depth)
{
    int value;

    value = get_vlc2(gb, vlc->table, vlc->bits, depth);

    /* stage-2, 3 bits exponent escape sequence */
    if (value-- == 0)
        value = get_bits (gb, get_bits (gb, 3) + 1);

    /* stage-3, optional */
    if (flag) {
        int tmp = vlc_stage3_values[value];

        if ((value & ~3) > 0)
            tmp += get_bits (gb, (value >> 2));
        value = tmp;
    }

    return value;
}
    """

    code = nor(str)
    print(code[0])
