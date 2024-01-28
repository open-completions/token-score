from spiral import ronin

from token_score import (
    Document,
    SyntaxToken,
    collect_identifiers,
    collect_syntax_tokens,
    compute_jaccard_similarity_score,
)


def test_collect_syntax_tokens_go():
    document = Document(
        lang="go",
        content=b'package hello_world\n\n// Hello World\nfunc main() {\na := "Hello World"}\n',
    )

    tree = document.parse()

    assert collect_syntax_tokens(
        tree,
        document.content,
    ) == [
        SyntaxToken(range=(0, 7), type="package"),
        SyntaxToken(range=(7, 8), type="unknown"),
        SyntaxToken(range=(8, 19), type="package_identifier"),
        SyntaxToken(range=(19, 21), type="\n"),
        SyntaxToken(range=(21, 35), type="comment"),
        SyntaxToken(range=(35, 36), type="unknown"),
        SyntaxToken(range=(36, 40), type="func"),
        SyntaxToken(range=(40, 41), type="unknown"),
        SyntaxToken(range=(41, 45), type="identifier"),
        SyntaxToken(range=(45, 46), type="("),
        SyntaxToken(range=(46, 47), type=")"),
        SyntaxToken(range=(47, 48), type="unknown"),
        SyntaxToken(range=(48, 49), type="{"),
        SyntaxToken(range=(49, 50), type="unknown"),
        SyntaxToken(range=(50, 51), type="identifier"),
        SyntaxToken(range=(51, 52), type="unknown"),
        SyntaxToken(range=(52, 54), type=":="),
        SyntaxToken(range=(54, 55), type="unknown"),
        SyntaxToken(range=(55, 56), type='"'),
        SyntaxToken(range=(56, 67), type="unknown"),
        SyntaxToken(range=(67, 68), type='"'),
        SyntaxToken(range=(68, 69), type="}"),
        SyntaxToken(range=(69, 70), type="\n"),
    ]


def test_collect_syntax_tokens_python():
    document = Document(
        lang="python", content=b'# Hello World\ndef main():\n\tabc = "abc"\n'
    )

    tree = document.parse()

    assert collect_syntax_tokens(
        tree,
        document.content,
    ) == [
        SyntaxToken(range=(0, 13), type="comment"),
        SyntaxToken(range=(13, 14), type="unknown"),
        SyntaxToken(range=(14, 17), type="def"),
        SyntaxToken(range=(17, 18), type="unknown"),
        SyntaxToken(range=(18, 22), type="identifier"),
        SyntaxToken(range=(22, 23), type="("),
        SyntaxToken(range=(23, 24), type=")"),
        SyntaxToken(range=(24, 25), type=":"),
        SyntaxToken(range=(25, 27), type="unknown"),
        SyntaxToken(range=(27, 30), type="identifier"),
        SyntaxToken(range=(30, 31), type="unknown"),
        SyntaxToken(range=(31, 32), type="="),
        SyntaxToken(range=(32, 33), type="unknown"),
        SyntaxToken(range=(33, 34), type='"'),
        SyntaxToken(range=(34, 37), type="unknown"),
        SyntaxToken(range=(37, 38), type='"'),
        SyntaxToken(range=(38, 39), type="unknown"),
    ]


def test_collect_syntax_tokens_java():
    document = Document(
        lang="java",
        content=b"package hello_world;\n\npublic class Main {\n\tpublic static void main(String[] args) {\n\t}\n}\n",
    )

    tree = document.parse()

    assert collect_syntax_tokens(
        tree,
        document.content,
    ) == [
        SyntaxToken(range=(0, 7), type="package"),
        SyntaxToken(range=(7, 8), type="unknown"),  # " "
        SyntaxToken(range=(8, 19), type="identifier"),
        SyntaxToken(range=(19, 20), type=";"),
        SyntaxToken(range=(20, 22), type="unknown"),  # "\n\n"
        SyntaxToken(range=(22, 28), type="public"),
        SyntaxToken(range=(28, 29), type="unknown"),  # " "
        SyntaxToken(range=(29, 34), type="class"),
        SyntaxToken(range=(34, 35), type="unknown"),  # " "
        SyntaxToken(range=(35, 39), type="identifier"),
        SyntaxToken(range=(39, 40), type="unknown"),  # " "
        SyntaxToken(range=(40, 41), type="{"),
        SyntaxToken(range=(41, 43), type="unknown"),  # "\n\t"
        SyntaxToken(range=(43, 49), type="public"),
        SyntaxToken(range=(49, 50), type="unknown"),  # " "
        SyntaxToken(range=(50, 56), type="static"),
        SyntaxToken(range=(56, 57), type="unknown"),  # " "
        SyntaxToken(range=(57, 61), type="void_type"),
        SyntaxToken(range=(61, 62), type="unknown"),  # " "
        SyntaxToken(range=(62, 66), type="identifier"),
        SyntaxToken(range=(66, 67), type="("),
        SyntaxToken(range=(67, 73), type="type_identifier"),
        SyntaxToken(range=(73, 74), type="["),
        SyntaxToken(range=(74, 75), type="]"),
        SyntaxToken(range=(75, 76), type="unknown"),  # " "
        SyntaxToken(range=(76, 80), type="identifier"),
        SyntaxToken(range=(80, 81), type=")"),
        SyntaxToken(range=(81, 82), type="unknown"),  # " "
        SyntaxToken(range=(82, 83), type="{"),
        SyntaxToken(range=(83, 85), type="unknown"),  # "\n\t"
        SyntaxToken(range=(85, 86), type="}"),
        SyntaxToken(range=(86, 87), type="unknown"),  # "\n"
        SyntaxToken(range=(87, 88), type="}"),
        SyntaxToken(range=(88, 89), type="unknown"),  # "\n"
    ]


def test_collect_syntax_tokens_javascript():
    document = Document(
        lang="javascript",
        content=b"function main() {\n    // My variable\n    let i = 10;\n}\n",
    )

    tree = document.parse()

    assert collect_syntax_tokens(
        tree,
        document.content,
    ) == [
        SyntaxToken(range=(0, 8), type="function"),
        SyntaxToken(range=(8, 9), type="unknown"),
        SyntaxToken(range=(9, 13), type="identifier"),
        SyntaxToken(range=(13, 14), type="("),
        SyntaxToken(range=(14, 15), type=")"),
        SyntaxToken(range=(15, 16), type="unknown"),
        SyntaxToken(range=(16, 17), type="{"),
        SyntaxToken(range=(17, 22), type="unknown"),
        SyntaxToken(range=(22, 36), type="comment"),
        SyntaxToken(range=(36, 41), type="unknown"),
        SyntaxToken(range=(41, 44), type="let"),
        SyntaxToken(range=(44, 45), type="unknown"),
        SyntaxToken(range=(45, 46), type="identifier"),
        SyntaxToken(range=(46, 47), type="unknown"),
        SyntaxToken(range=(47, 48), type="="),
        SyntaxToken(range=(48, 49), type="unknown"),
        SyntaxToken(range=(49, 51), type="number"),
        SyntaxToken(range=(51, 52), type=";"),
        SyntaxToken(range=(52, 53), type="unknown"),
        SyntaxToken(range=(53, 54), type="}"),
        SyntaxToken(range=(54, 55), type="unknown"),
    ]


def test_collect_syntax_tokens_cpp():
    document = Document(
        lang="c++",
        content=b'#include <iostream>\n\nint main() {\n\tstd::cout << "Hello World!";\n\treturn 0;\n}\n',
    )

    tree = document.parse()

    assert collect_syntax_tokens(
        tree,
        document.content,
    ) == [
        SyntaxToken(range=(0, 8), type="#include"),
        SyntaxToken(range=(8, 9), type="unknown"),
        SyntaxToken(range=(9, 19), type="system_lib_string"),
        SyntaxToken(range=(19, 21), type="\n"),
        SyntaxToken(range=(21, 24), type="primitive_type"),
        SyntaxToken(range=(24, 25), type="unknown"),
        SyntaxToken(range=(25, 29), type="identifier"),
        SyntaxToken(range=(29, 30), type="("),
        SyntaxToken(range=(30, 31), type=")"),
        SyntaxToken(range=(31, 32), type="unknown"),
        SyntaxToken(range=(32, 33), type="{"),
        SyntaxToken(range=(33, 35), type="unknown"),
        SyntaxToken(range=(35, 38), type="namespace_identifier"),
        SyntaxToken(range=(38, 40), type="::"),
        SyntaxToken(range=(40, 44), type="identifier"),
        SyntaxToken(range=(44, 45), type="unknown"),
        SyntaxToken(range=(45, 47), type="<<"),
        SyntaxToken(range=(47, 48), type="unknown"),
        SyntaxToken(range=(48, 49), type='"'),
        SyntaxToken(range=(49, 61), type="unknown"),
        SyntaxToken(range=(61, 62), type='"'),
        SyntaxToken(range=(62, 63), type=";"),
        SyntaxToken(range=(63, 65), type="unknown"),
        SyntaxToken(range=(65, 71), type="return"),
        SyntaxToken(range=(71, 72), type="unknown"),
        SyntaxToken(range=(72, 73), type="number_literal"),
        SyntaxToken(range=(73, 74), type=";"),
        SyntaxToken(range=(74, 75), type="unknown"),
        SyntaxToken(range=(75, 76), type="}"),
        SyntaxToken(range=(76, 77), type="unknown"),
    ]


def test_collect_identifiers_go():
    document = Document(
        lang="go",
        content=b'package hello_world\n\n// Hello World\nfunc main() {\na := "Hello World"}\ntype Hello struct {\n\tint field\n}',
    )

    tree = document.parse()

    ids = collect_identifiers(tree, document)

    assert ids == [
        SyntaxToken(range=(8, 19), type="package_identifier"),
        SyntaxToken(range=(41, 45), type="identifier"),
        SyntaxToken(range=(50, 51), type="identifier"),
        SyntaxToken(range=(75, 80), type="type_identifier"),
        SyntaxToken(range=(91, 94), type="field_identifier"),
        SyntaxToken(range=(95, 100), type="type_identifier"),
    ]

    assert [document.token_to_bytes(id) for id in ids] == [
        b"hello_world",
        b"main",
        b"a",
        b"Hello",
        b"int",
        b"field",
    ]


def test_collect_identifiers_python():
    document = Document(
        lang="python", content=b'from abc import bcd\n\ndef main():\n\tabc = "abc"\n'
    )

    tree = document.parse()

    ids = collect_identifiers(tree, document)

    assert ids == [
        SyntaxToken(range=(5, 8), type="identifier"),
        SyntaxToken(range=(16, 19), type="identifier"),
        SyntaxToken(range=(25, 29), type="identifier"),
        SyntaxToken(range=(34, 37), type="identifier"),
    ]

    assert [document.token_to_bytes(id) for id in ids] == [
        b"abc",
        b"bcd",
        b"main",
        b"abc",
    ]


def test_collect_identifiers_java():
    document = Document(
        lang="java",
        content=b"class HelloWorld {\n    public static void main(String[] args) {\n        int number = 42;\n    }\n}",
    )

    tree = document.parse()

    ids = collect_identifiers(tree, document)

    assert ids == [
        SyntaxToken(range=(6, 16), type="identifier"),
        SyntaxToken(range=(42, 46), type="identifier"),
        SyntaxToken(range=(47, 53), type="type_identifier"),
        SyntaxToken(range=(56, 60), type="identifier"),
        SyntaxToken(range=(76, 82), type="identifier"),
    ]

    assert [document.token_to_bytes(id) for id in ids] == [
        b"HelloWorld",
        b"main",
        b"String",
        b"args",
        b"number",
    ]


def test_collect_identifiers_javascript():
    document = Document(
        lang="javascript",
        content=b"class Hello {}\n function main() {\n    let i = console.log();\n}\n",
    )

    tree = document.parse()

    ids = collect_identifiers(tree, document)

    assert ids == [
        SyntaxToken(range=(6, 11), type="identifier"),
        SyntaxToken(range=(25, 29), type="identifier"),
        SyntaxToken(range=(42, 43), type="identifier"),
        SyntaxToken(range=(46, 53), type="identifier"),
        SyntaxToken(range=(54, 57), type="property_identifier"),
    ]

    assert [document.token_to_bytes(id) for id in ids] == [
        b"Hello",
        b"main",
        b"i",
        b"console",
        b"log",
    ]


def test_collect_identifiers_cpp():
    document = Document(
        lang="c++",
        content=b"#include <iostream>\n\nint main() {\n\tstd::cout << myvar;\n};\nclass Hello {int x;};\n",
    )

    tree = document.parse()

    ids = collect_identifiers(tree, document)

    assert ids == [
        SyntaxToken(range=(25, 29), type="identifier"),
        SyntaxToken(range=(35, 38), type="namespace_identifier"),
        SyntaxToken(range=(40, 44), type="identifier"),
        SyntaxToken(range=(48, 53), type="identifier"),
        SyntaxToken(range=(64, 69), type="type_identifier"),
        SyntaxToken(range=(75, 76), type="field_identifier"),
    ]

    assert [document.token_to_bytes(id) for id in ids] == [
        b"main",
        b"std",
        b"cout",
        b"myvar",
        b"Hello",
        b"x",
    ]


def test_compute_jaccard_similarity_score():
    assert (
        compute_jaccard_similarity_score(set(["a", "b", "c"]), set(["a", "b", "c"]))
        == 1.0
    )

    spiral_outputs = [["user", "Count"], ["xml", "Form", "Template"]]
    tokenizer_outputs = [["user", "Count"], ["x", "ml", "Form", "Template"]]
    scores = []

    for spiral_set, tokenizer_set in zip(spiral_outputs, tokenizer_outputs):
        scores.append(
            compute_jaccard_similarity_score(set(spiral_set), set(tokenizer_set))
        )

    assert scores == [1.0, 0.4]


def test_spiral_usage():
    expected = [
        ["m", "Start", "C", "Data"],
        ["nonnegative", "decimal", "type"],
        ["get", "Utf8", "Octets"],
        ["GPS", "module"],
        ["save", "file", "as"],
        ["nbr", "Of", "bugs"],
    ]
    actual = []

    for s in [
        "mStartCData",
        "nonnegativedecimaltype",
        "getUtf8Octets",
        "GPSmodule",
        "savefileas",
        "nbrOfbugs",
    ]:
        actual.append(ronin.split(s))

    assert actual == expected

    assert ronin.split("snake_case") == ["snake", "case"]
    assert ronin.split("InvalidCamel_CaseName") == ["Invalid", "Camel", "Case", "Name"]
    assert ronin.split("a space") == ["a", "space"]
