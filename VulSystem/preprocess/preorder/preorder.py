from tree_sitter import Language, Parser
from preprocess.preorder.prepare_data import get_sequences


def parse_ast(source):
    # C_LANGUAGE = Language('build_languages/my-languages.so', 'c')
    CPP_LANGUAGE = Language('../build_languages/my-languages.so', 'cpp')
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)  # set the parser for certain language
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    return tree


def trans_to_sequences(ast):
    sequence = []
    get_sequences(ast, sequence)
    path = [token.decode('utf-8') if type(token) is bytes else token for token in sequence]
    print(len(path))
    str = ' '.join(path)
    # print(str)
    return str


if __name__ == '__main__':
    str = r"""
    static void goodG2B2()
{
    char * data;
    char dataBuffer[100] = "";
    data = dataBuffer;
    if(5==5)
    {
        /* FIX: get the hostname from a string literal */
        strcpy(data, "hostname");
    }
    /* POTENTIAL FLAW: set the hostname to data obtained from a potentially external source */
    if (!SetComputerNameA(data))
    {
        printLine("Failure setting computer name");
        exit(1);
    }
}
    """
    tree = parse_ast(str)
    print(trans_to_sequences(tree))
