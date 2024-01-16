from token_score import (
    Document,
    DocumentSemanticToken,
    collect_semantic_tokens,
)


def test_collect_semantic_tokens_go():
    document = Document(
        lang="go",
        content=b'package hello_world\n\n// Hello World\nfunc main() {\na := "Hello World"}\n',
    )

    tree = document.parse()

    assert collect_semantic_tokens(
        tree,
        document.content,
    ) == [
        DocumentSemanticToken(range=(0, 7), type="package"),
        DocumentSemanticToken(range=(7, 8), type="whitespace"),
        DocumentSemanticToken(range=(8, 19), type="package_identifier"),
        DocumentSemanticToken(range=(19, 21), type="\n"),
        DocumentSemanticToken(range=(21, 35), type="comment"),
        DocumentSemanticToken(range=(35, 36), type="whitespace"),
        DocumentSemanticToken(range=(36, 40), type="func"),
        DocumentSemanticToken(range=(40, 41), type="whitespace"),
        DocumentSemanticToken(range=(41, 45), type="identifier"),
        DocumentSemanticToken(range=(45, 46), type="("),
        DocumentSemanticToken(range=(46, 47), type=")"),
        DocumentSemanticToken(range=(47, 48), type="whitespace"),
        DocumentSemanticToken(range=(48, 49), type="{"),
        DocumentSemanticToken(range=(49, 50), type="whitespace"),
        DocumentSemanticToken(range=(50, 51), type="identifier"),
        DocumentSemanticToken(range=(51, 52), type="whitespace"),
        DocumentSemanticToken(range=(52, 54), type=":="),
        DocumentSemanticToken(range=(54, 55), type="whitespace"),
        DocumentSemanticToken(range=(55, 56), type='"'),
        DocumentSemanticToken(range=(56, 67), type="whitespace"),
        DocumentSemanticToken(range=(67, 68), type='"'),
        DocumentSemanticToken(range=(68, 69), type="}"),
        DocumentSemanticToken(range=(69, 70), type="\n"),
    ]


def test_collect_semantic_tokens_python():
    document = Document(
        lang="python", content=b'# Hello World\ndef main():\n\tabc = "abc"\n'
    )

    tree = document.parse()

    assert collect_semantic_tokens(
        tree,
        document.content,
    ) == [
        DocumentSemanticToken(range=(0, 13), type="comment"),
        DocumentSemanticToken(range=(13, 14), type="whitespace"),
        DocumentSemanticToken(range=(14, 17), type="def"),
        DocumentSemanticToken(range=(17, 18), type="whitespace"),
        DocumentSemanticToken(range=(18, 22), type="identifier"),
        DocumentSemanticToken(range=(22, 23), type="("),
        DocumentSemanticToken(range=(23, 24), type=")"),
        DocumentSemanticToken(range=(24, 25), type=":"),
        DocumentSemanticToken(range=(25, 27), type="whitespace"),
        DocumentSemanticToken(range=(27, 30), type="identifier"),
        DocumentSemanticToken(range=(30, 31), type="whitespace"),
        DocumentSemanticToken(range=(31, 32), type="="),
        DocumentSemanticToken(range=(32, 33), type="whitespace"),
        DocumentSemanticToken(range=(33, 34), type='"'),
        DocumentSemanticToken(range=(34, 37), type="whitespace"),
        DocumentSemanticToken(range=(37, 38), type='"'),
        DocumentSemanticToken(range=(38, 39), type="whitespace"),
    ]


def test_collect_semantic_tokens_java():
    document = Document(
        lang="java",
        content=b"package hello_world;\n\npublic class Main {\n\tpublic static void main(String[] args) {\n\t}\n}\n",
    )

    tree = document.parse()

    assert collect_semantic_tokens(
        tree,
        document.content,
    ) == [
        DocumentSemanticToken(range=(0, 7), type="package"),
        DocumentSemanticToken(range=(7, 8), type="whitespace"),  # " "
        DocumentSemanticToken(range=(8, 19), type="identifier"),
        DocumentSemanticToken(range=(19, 20), type=";"),
        DocumentSemanticToken(range=(20, 22), type="whitespace"),  # "\n\n"
        DocumentSemanticToken(range=(22, 28), type="public"),
        DocumentSemanticToken(range=(28, 29), type="whitespace"),  # " "
        DocumentSemanticToken(range=(29, 34), type="class"),
        DocumentSemanticToken(range=(34, 35), type="whitespace"),  # " "
        DocumentSemanticToken(range=(35, 39), type="identifier"),
        DocumentSemanticToken(range=(39, 40), type="whitespace"),  # " "
        DocumentSemanticToken(range=(40, 41), type="{"),
        DocumentSemanticToken(range=(41, 43), type="whitespace"),  # "\n\t"
        DocumentSemanticToken(range=(43, 49), type="public"),
        DocumentSemanticToken(range=(49, 50), type="whitespace"),  # " "
        DocumentSemanticToken(range=(50, 56), type="static"),
        DocumentSemanticToken(range=(56, 57), type="whitespace"),  # " "
        DocumentSemanticToken(range=(57, 61), type="void_type"),
        DocumentSemanticToken(range=(61, 62), type="whitespace"),  # " "
        DocumentSemanticToken(range=(62, 66), type="identifier"),
        DocumentSemanticToken(range=(66, 67), type="("),
        DocumentSemanticToken(range=(67, 73), type="type_identifier"),
        DocumentSemanticToken(range=(73, 74), type="["),
        DocumentSemanticToken(range=(74, 75), type="]"),
        DocumentSemanticToken(range=(75, 76), type="whitespace"),  # " "
        DocumentSemanticToken(range=(76, 80), type="identifier"),
        DocumentSemanticToken(range=(80, 81), type=")"),
        DocumentSemanticToken(range=(81, 82), type="whitespace"),  # " "
        DocumentSemanticToken(range=(82, 83), type="{"),
        DocumentSemanticToken(range=(83, 85), type="whitespace"),  # "\n\t"
        DocumentSemanticToken(range=(85, 86), type="}"),
        DocumentSemanticToken(range=(86, 87), type="whitespace"),  # "\n"
        DocumentSemanticToken(range=(87, 88), type="}"),
        DocumentSemanticToken(range=(88, 89), type="whitespace"),  # "\n"
    ]


def test_collect_semantic_tokens_javascript():
    document = Document(
        lang="javascript",
        content=b"function main() {\n    // My variable\n    let i = 10;\n}\n",
    )

    tree = document.parse()

    assert collect_semantic_tokens(
        tree,
        document.content,
    ) == [
        DocumentSemanticToken(range=(0, 8), type="function"),
        DocumentSemanticToken(range=(8, 9), type="whitespace"),
        DocumentSemanticToken(range=(9, 13), type="identifier"),
        DocumentSemanticToken(range=(13, 14), type="("),
        DocumentSemanticToken(range=(14, 15), type=")"),
        DocumentSemanticToken(range=(15, 16), type="whitespace"),
        DocumentSemanticToken(range=(16, 17), type="{"),
        DocumentSemanticToken(range=(17, 22), type="whitespace"),
        DocumentSemanticToken(range=(22, 36), type="comment"),
        DocumentSemanticToken(range=(36, 41), type="whitespace"),
        DocumentSemanticToken(range=(41, 44), type="let"),
        DocumentSemanticToken(range=(44, 45), type="whitespace"),
        DocumentSemanticToken(range=(45, 46), type="identifier"),
        DocumentSemanticToken(range=(46, 47), type="whitespace"),
        DocumentSemanticToken(range=(47, 48), type="="),
        DocumentSemanticToken(range=(48, 49), type="whitespace"),
        DocumentSemanticToken(range=(49, 51), type="number"),
        DocumentSemanticToken(range=(51, 52), type=";"),
        DocumentSemanticToken(range=(52, 53), type="whitespace"),
        DocumentSemanticToken(range=(53, 54), type="}"),
        DocumentSemanticToken(range=(54, 55), type="whitespace"),
    ]


def test_collect_semantic_tokens_cpp():
    document = Document(
        lang="c++",
        content=b'#include <iostream>\n\nint main() {\n\tstd::cout << "Hello World!";\n\treturn 0;\n}\n',
    )

    tree = document.parse()

    assert collect_semantic_tokens(
        tree,
        document.content,
    ) == [
        DocumentSemanticToken(range=(0, 8), type="#include"),
        DocumentSemanticToken(range=(8, 9), type="whitespace"),
        DocumentSemanticToken(range=(9, 19), type="system_lib_string"),
        DocumentSemanticToken(range=(19, 21), type="\n"),
        DocumentSemanticToken(range=(21, 24), type="primitive_type"),
        DocumentSemanticToken(range=(24, 25), type="whitespace"),
        DocumentSemanticToken(range=(25, 29), type="identifier"),
        DocumentSemanticToken(range=(29, 30), type="("),
        DocumentSemanticToken(range=(30, 31), type=")"),
        DocumentSemanticToken(range=(31, 32), type="whitespace"),
        DocumentSemanticToken(range=(32, 33), type="{"),
        DocumentSemanticToken(range=(33, 35), type="whitespace"),
        DocumentSemanticToken(range=(35, 38), type="namespace_identifier"),
        DocumentSemanticToken(range=(38, 40), type="::"),
        DocumentSemanticToken(range=(40, 44), type="identifier"),
        DocumentSemanticToken(range=(44, 45), type="whitespace"),
        DocumentSemanticToken(range=(45, 47), type="<<"),
        DocumentSemanticToken(range=(47, 48), type="whitespace"),
        DocumentSemanticToken(range=(48, 49), type='"'),
        DocumentSemanticToken(range=(49, 61), type="whitespace"),
        DocumentSemanticToken(range=(61, 62), type='"'),
        DocumentSemanticToken(range=(62, 63), type=";"),
        DocumentSemanticToken(range=(63, 65), type="whitespace"),
        DocumentSemanticToken(range=(65, 71), type="return"),
        DocumentSemanticToken(range=(71, 72), type="whitespace"),
        DocumentSemanticToken(range=(72, 73), type="number_literal"),
        DocumentSemanticToken(range=(73, 74), type=";"),
        DocumentSemanticToken(range=(74, 75), type="whitespace"),
        DocumentSemanticToken(range=(75, 76), type="}"),
        DocumentSemanticToken(range=(76, 77), type="whitespace"),
    ]
