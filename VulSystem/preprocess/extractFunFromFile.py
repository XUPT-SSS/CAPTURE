from tree_sitter import Language, Parser
from preprocess.lang_processors.cpp_processor import CppProcessor
import pandas as pd

def selectFun(nodes, functions: list, code, flatten):
    """
    递归的从AST中抽取所有函数
    :param nodes:  AST节点
    :param functions: 函数list集合
    :return:
    """
    if len(nodes.children) == 0:
        return
    for child_node in nodes.children:
        if child_node.type == "function_definition":
            function_start_line = child_node.start_point[0]
            function_end_line = child_node.end_point[0]
            # 不在同一行
            if function_start_line != function_end_line:
                function_code = code[function_start_line:function_end_line + 1]
                if flatten:
                    function_code = " ".join(function_code)
                else:
                    function_code = "\n".join(function_code)
            else:
                function_code = code[function_start_line]
            functions.append(function_code)
        #comment 6.12
        selectFun(child_node, functions, code, flatten)


def extract(path: str, language, flatten=False):
    """
    抽取文件中所有函数
    :param path: 文件所在路径
    :return: 所有函数list集合
    """
    #CPP_LANGUAGE = Language('./languages.so', 'cpp')
    # LANGUAGEs = Language('C:\\Users\\Administrator\\PycharmProjects\\VulSystem\\preprocess\\languages.so', language)


    CPP_LANGUAGE = Language('/data/rqiu/multimodal_rerprogramming/VulSystem/preprocess/languages.so','cpp')
    LANGUAGEs = Language('/data/rqiu/multimodal_rerprogramming/VulSystem/preprocess/languages.so', language)
    parser = Parser()
    parser.set_language(LANGUAGEs)
    with open(path) as f:
        codes = f.readlines()

    code = ''
    for line in codes:
        code += line
    #comment
    # tree = parser.parse(code.encode('utf-8').decode('unicode_escape').encode())
    tree = parser.parse(bytes(code,"utf-8"))
    root_node = tree.root_node
    functions = []
    # 为了确定起始行
    code = code.split("\n")
    # 保留原格式还是将其展平
    flatten = flatten
    selectFun(root_node, functions, code, flatten)
    return functions

# def extract(code, language, flatten=False):
#     test
#     """
#     抽取文件中所有函数
#     :param path: 文件所在路径
#     :return: 所有函数list集合
#     """
#     #CPP_LANGUAGE = Language('./languages.so', 'cpp')
#     # LANGUAGEs = Language('C:\\Users\\Administrator\\PycharmProjects\\VulSystem\\preprocess\\languages.so', language)
#
#
#     #CPP_LANGUAGE = Language('D:\\data_process\VulSystem\\preprocess\\languages.so','cpp')
#     LANGUAGEs = Language('D:\\data_process\\VulSystem\\preprocess\\languages.so', language)
#     parser = Parser()
#     parser.set_language(LANGUAGEs)
#
#     tree = parser.parse(code.encode('utf-8').decode('unicode_escape').encode())
#     root_node = tree.root_node
#     functions = []
#     # 为了确定起始行
#     code = code.split("\n")
#     # 保留原格式还是将其展平
#     flatten = flatten
#     selectFun(root_node, functions, code, flatten)
#     return functions

def dfs(root_node):
    if  not root_node:
        return


if __name__ == '__main__':

    # funs = extract('./a.cpp', 'cpp')
    # print(funs)
    # print(type(funs))
    # cpp_processor = CppProcessor()
    #
    # for fun in funs:
    #     fun = cpp_processor.tokenize_code(fun)
    #     print(fun)




    # file_name="/data/rqiu/multimodal_rerprogramming/VulSystem/test.csv"
    # dataframe = pd.read_csv(file_name)
    # funs = dataframe['text']
    # cpp_processor = CppProcessor(root_folder='/data/rqiu/multimodal_rerprogramming/VulSystem/')
    # count = 0
    # for fun in funs:
    #     fun = cpp_processor.tokenize_code(fun)
    #     count+=1
    #     print(fun)
    #     print(len(fun))
    #     if count==1:
    #         break


    file_name="/data/rqiu/multimodal_rerprogramming/VulSystem/test.csv"
    dataframe = pd.read_csv(file_name)
    funs = dataframe['text']
    func_demo = funs[0]
    print(func_demo)
    print(type(func_demo))
    CPP_LANGUAGE = Language('./languages.so', 'cpp')
    cpp_parser = Parser()
    cpp_parser.set_language(CPP_LANGUAGE)


    tree = cpp_parser.parse(bytes(func_demo, "utf8"))
    print(tree)
    root_node = tree.root_node



