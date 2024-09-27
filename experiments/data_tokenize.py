import pandas as pd
from tree_sitter import Language, Parser
import collections
import pprint
import sys
import graphviz
lib_path = '../VulSystem/preprocess/languages.so'
language = Language(lib_path, 'cpp')
parser = Parser()
parser.set_language(language)

# from normalization import nor

sys.path.append(r'/data/rqiu/multimodal_rerprogramming/VulSystem')
from normalization import *

class Tokenize():
    '''
    1.token init
            token_table                         --->                       nomalization                                        ---token_idx
        ['NS_IMETHODIMP', 'nsGlobalWindow', '::', 'SetScreenY'] ---> ['static VAR1 FUN1 ( VAR2 * VAR3 , VAR4 * VAR5 ) ]   [(unk,0), (VAR1,1),(VAR2,2)...]

        struct_info_table                                                                                                  ---struct_idx
        ['translation_unit', 'function_definition', 'function_declarator', 'qualified_identifier', 'parameter_list', 'parameter_declaration',
          1                            2                      3                  4
    '''
    def __init__(self,min_freq=0,reserved_tokens=None):
        self.min_freq = min_freq
        d2a_file_name = '../dataset/d2a/d2a_train.csv'
        vuldeepecker_file_name = '../dataset/vuldeepecker/vuldeepecker_undersampling.csv'
        vul_file_name = '../dataset/vul/train.csv'
        reveal_file_name = '../dataset/reveal/reveal_train.csv'
        our_file_name = '../dataset/our/our_train.csv'
        devign_file_name = '../dataset/devign/devign_train.csv'
        dataframe = pd.read_csv(our_file_name)

        funs = dataframe['text']
        count = 0
        self.type_list =[]
        self.value_list = []

        ast = AST()
        for func in funs:
            func = nor(func)[0]
            _type_list,_value_list = ast.get_ast_type_value(func)
            self.type_list += _type_list
            self.value_list+= _value_list
            # if count >= 500:
            #     break
            count += 1
        counter_type = count_corpus(self.type_list)
        counter_value = count_corpus(self.value_list)
        self._type_freqs = sorted(counter_type.items(), key=lambda x: x[1],reverse=True)
        self._value_freqs = sorted(counter_value.items(), key=lambda x: x[1], reverse=True)

        #GO s1 EOS SEP GO s2 EOS
        UNK = ['<UNK>']
        GO = ['<GO>']
        EOS = ['<EOS>']
        SEP = ['<SEP>']
        PAD = ['<PAD>']
        self.vocab_list = UNK + GO + EOS + SEP + PAD
        self.vocab_table = {token: idx
            for idx, token in enumerate(self.vocab_list)}
        self._vocab_freq = self._type_freqs + self._value_freqs
        for token, freq in self._vocab_freq:
            if token not in self.vocab_table:
                self.vocab_list.append(token)
                self.vocab_table[token] = len(self.vocab_list) - 1

        print(self.vocab_table)
        print(len(self.vocab_table))

    def tokenize(self, *func, block_size=576,value_size=400,type_size=171):
        ast = AST()
        func = ''.join(func)
        func = nor(func)[0]
        # print(func)
        ast_type_list,ast_value_list = ast.get_ast_type_value(func)
        len_value = len(ast_value_list)
        len_type = len(ast_type_list)


        if(len_type < type_size ):
            ast_type_list = ['<GO>'] + ast_type_list[0:len_type] + ['<EOS>'] + ['<SEP>']
        else:
            ast_type_list = ['<GO>'] + ast_type_list[0:type_size] + ['<EOS>'] + ['<SEP>']

        if( len_value < block_size - len(ast_type_list) -2 ):
            ast_value_list = ['<GO>'] + ast_value_list[0:len_value] + ['<EOS>']
        else:
            ast_value_list = ['<GO>'] + ast_value_list[0:value_size] + ['<EOS>']

        token_list = ast_type_list + ast_value_list
        if (len(token_list)) < block_size:
            for i in range(len(token_list),block_size):
                token_list.append('<PAD>')
        transmit_func_to_dict = {}

        _token_ids = [0 for i in range(block_size)]

        idx = 0
        for elem in token_list:
            if elem in self.vocab_table :
                _token_ids[idx] = self.vocab_table[elem]
            else:
                _token_ids[idx] = 0
            idx += 1
        transmit_func_to_dict['type_ids'] = _token_ids


        return transmit_func_to_dict
    def tokenize_v2(self, *func, block_size=576,value_size=400,type_size=171):
        ast = AST()
        func = ''.join(func)
        func = nor(func)[0]
        # print(func)
        ast_type_list,ast_value_list = ast.get_ast_type_value(func)
        len_value = len(ast_value_list)
        len_type = len(ast_type_list)

        if(len_type < block_size - 2 ):
            ast_type_list = ['<GO>'] + ast_type_list[0:len_type] + ['<EOS>']
        else:
            ast_type_list = ['<GO>'] + ast_type_list[0:block_size - 2 ] + ['<EOS>']

        if( len_value < block_size - 2 ):
            ast_value_list = ['<GO>'] + ast_value_list[0:len_value] + ['<EOS>']
        else:
            ast_value_list = ['<GO>'] + ast_value_list[0:block_size - 2 ] + ['<EOS>']


        if (len(ast_type_list)) < block_size:
            for i in range(len(ast_type_list),block_size):
                ast_type_list.append('<PAD>')
        if (len(ast_value_list)) < block_size:
            for i in range(len(ast_value_list),block_size):
                ast_value_list.append('<PAD>')
        transmit_func_to_dict = {}

        _type_ids = [0 for i in range(0,block_size)]
        _value_ids = [0 for i in range(0,block_size)]
        idx = 0
        for elem in ast_type_list:
            if elem in self.vocab_table :
                _type_ids[idx] = self.vocab_table[elem]
            else:
                _type_ids[idx] = 0
            idx += 1
        idx = 0
        for elem in ast_value_list:
            if elem in self.vocab_table :
                _value_ids[idx] = self.vocab_table[elem]
            else:
                _value_ids[idx] = 0
            idx += 1
        transmit_func_to_dict['type_ids'] = _type_ids
        transmit_func_to_dict['value_ids'] = _value_ids


        return transmit_func_to_dict

    def index_to_token(self,indices):
        '''
        :param index:
        :return:
        '''
        if not isinstance(indices, (list, tuple)):
            return self.type_list[indices]
        return [self.type_list[index] for index in indices],[self.value_list[index] for index in indices]


    def get_vocab(self):
        '''
        :param self:
        :return: vocab_size
        '''
        # return max(len(self.type_table),len(self.value_table))
        return len(self.vocab_table)

#Statistics token frequency
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for token in tokens ]
    return collections.Counter(tokens)


#Get: ast path  and token
class AST():
    def get_token(self,func):
        func = ''.join(func)
        tree = parser.parse(bytes(func, "utf8"))
        root_node = tree.root_node
        tokens_index = AST.ast_node_position(self,root_node,0)
        cpp_loc = func.split('\n')
        code_tokens = [AST.node_position_to_token(self,x, cpp_loc) for x in tokens_index]
        return code_tokens
        # print(token_index)
        # token = AST.node_position_to_token(self,token_index,func)
        # print(token)

    def get_ast(self,func):
        func = ''.join(func)
        tree = parser.parse(bytes(func, "utf8"))
        root_node = tree.root_node
        token_list = AST.preorder_ast(self,root_node, 0)
        return token_list
    def get_ast_type_value(self,func):
        tree = parser.parse(bytes(func, "utf8"))
        root_node = tree.root_node
        type_list,value_list = AST.preorder(self,root_node, 0)
        return type_list,value_list
    def ast_node_position(self,root_node,depth):
        depth+=1
        if depth > 50:
            return[]
        '''
        :param root_node:
        :return: get every token start and end point  [((0, 0), (0, 13)), ((1, 0), (1, 14)), ((1, 14), (1, 16)), ((1, 16), (1, 26)), ((1, 26), (1, 27)),
        '''
        if (len(root_node.children) == 0 or root_node.type.find('string') != -1) and root_node.type != 'comment':
            return [(root_node.start_point, root_node.end_point)]
        else:
            code_tokens = []
            for child in root_node.children:
                code_tokens += AST.ast_node_position(self,child,depth)
            return code_tokens
    def node_position_to_token(self,index,code):
        '''
        :param index: index from   ast_tokenidx
        :param code: single func
        :return: ['NS_IMETHODIMP', 'nsGlobalWindow', '::', 'SetScreenY', '(', 'PRInt32', 'aScreenY', ')', '{', 'FORWARD_TO_OUTER', '(', 'SetScreenY', ',', '(', 'aScreenY', ')']
        '''
        start_point = index[0]
        end_point = index[1]
        if start_point[0] == end_point[0]:
            s = code[start_point[0]][start_point[1]:end_point[1]]
        else:
            s = ""
            s += code[start_point[0]][start_point[1]:]
            for i in range(start_point[0] + 1, end_point[0]):
                s += code[i]
            s += code[end_point[0]][:end_point[1]]
        return s

    def preorder_ast(self,root_node,depth):
        depth += 1
        if depth > 50:
            return []
        if root_node == None:
            return []
        token_list = []
        if (root_node.child_count != 0):
            token_list = [root_node.type]
        if (root_node.child_count == 0):
            token_list = [root_node.text]
        for leaf in root_node.children:
            token_list += AST.preorder(self,leaf, depth)
        return token_list

    def preorder(self,root_node, depth):
        depth += 1
        if depth > 50:
            return [],[]
        if root_node == None:
            return [],[]
        type_list,value_list = [],[]
        if (root_node.child_count != 0):
            # type_list,value_list = [root_node.type],[root_node.type+'Value']
            type_list, value_list = [root_node.type], []
        if (root_node.child_count == 0):
            # type_list, value_list = [root_node.type],[root_node.text]
            type_list, value_list = [], [root_node.text]
        for leaf in root_node.children:
            type_list_ret,value_list_ret = AST.preorder(self,leaf, depth)
            type_list  += type_list_ret
            value_list += value_list_ret
        return type_list,value_list

    def call_able(self,num):
        print(num)

def main():
    dataframe = pd.read_csv('../dataset/d2a/d2a_train.csv')
    #func = dataframe['text'][209]
    func = """
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
    # print(func)
    print()
    ast = AST()
    token = ast.get_ast_type_value(nor(func)[0])
    print(token)
    ##print(nor(func))
if __name__ == '__main__':
    main()
