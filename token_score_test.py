from spiral import ronin

from token_score import (
    Document,
    SemanticToken,
    collect_identifiers,
    collect_semantic_tokens,
    compute_jaccard_similarity_score,
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
        SemanticToken(range=(0, 7), type="package"),
        SemanticToken(range=(7, 8), type="whitespace"),
        SemanticToken(range=(8, 19), type="package_identifier"),
        SemanticToken(range=(19, 21), type="\n"),
        SemanticToken(range=(21, 35), type="comment"),
        SemanticToken(range=(35, 36), type="whitespace"),
        SemanticToken(range=(36, 40), type="func"),
        SemanticToken(range=(40, 41), type="whitespace"),
        SemanticToken(range=(41, 45), type="identifier"),
        SemanticToken(range=(45, 46), type="("),
        SemanticToken(range=(46, 47), type=")"),
        SemanticToken(range=(47, 48), type="whitespace"),
        SemanticToken(range=(48, 49), type="{"),
        SemanticToken(range=(49, 50), type="whitespace"),
        SemanticToken(range=(50, 51), type="identifier"),
        SemanticToken(range=(51, 52), type="whitespace"),
        SemanticToken(range=(52, 54), type=":="),
        SemanticToken(range=(54, 55), type="whitespace"),
        SemanticToken(range=(55, 56), type='"'),
        SemanticToken(range=(56, 67), type="whitespace"),
        SemanticToken(range=(67, 68), type='"'),
        SemanticToken(range=(68, 69), type="}"),
        SemanticToken(range=(69, 70), type="\n"),
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
        SemanticToken(range=(0, 13), type="comment"),
        SemanticToken(range=(13, 14), type="whitespace"),
        SemanticToken(range=(14, 17), type="def"),
        SemanticToken(range=(17, 18), type="whitespace"),
        SemanticToken(range=(18, 22), type="identifier"),
        SemanticToken(range=(22, 23), type="("),
        SemanticToken(range=(23, 24), type=")"),
        SemanticToken(range=(24, 25), type=":"),
        SemanticToken(range=(25, 27), type="whitespace"),
        SemanticToken(range=(27, 30), type="identifier"),
        SemanticToken(range=(30, 31), type="whitespace"),
        SemanticToken(range=(31, 32), type="="),
        SemanticToken(range=(32, 33), type="whitespace"),
        SemanticToken(range=(33, 34), type='"'),
        SemanticToken(range=(34, 37), type="whitespace"),
        SemanticToken(range=(37, 38), type='"'),
        SemanticToken(range=(38, 39), type="whitespace"),
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
        SemanticToken(range=(0, 7), type="package"),
        SemanticToken(range=(7, 8), type="whitespace"),  # " "
        SemanticToken(range=(8, 19), type="identifier"),
        SemanticToken(range=(19, 20), type=";"),
        SemanticToken(range=(20, 22), type="whitespace"),  # "\n\n"
        SemanticToken(range=(22, 28), type="public"),
        SemanticToken(range=(28, 29), type="whitespace"),  # " "
        SemanticToken(range=(29, 34), type="class"),
        SemanticToken(range=(34, 35), type="whitespace"),  # " "
        SemanticToken(range=(35, 39), type="identifier"),
        SemanticToken(range=(39, 40), type="whitespace"),  # " "
        SemanticToken(range=(40, 41), type="{"),
        SemanticToken(range=(41, 43), type="whitespace"),  # "\n\t"
        SemanticToken(range=(43, 49), type="public"),
        SemanticToken(range=(49, 50), type="whitespace"),  # " "
        SemanticToken(range=(50, 56), type="static"),
        SemanticToken(range=(56, 57), type="whitespace"),  # " "
        SemanticToken(range=(57, 61), type="void_type"),
        SemanticToken(range=(61, 62), type="whitespace"),  # " "
        SemanticToken(range=(62, 66), type="identifier"),
        SemanticToken(range=(66, 67), type="("),
        SemanticToken(range=(67, 73), type="type_identifier"),
        SemanticToken(range=(73, 74), type="["),
        SemanticToken(range=(74, 75), type="]"),
        SemanticToken(range=(75, 76), type="whitespace"),  # " "
        SemanticToken(range=(76, 80), type="identifier"),
        SemanticToken(range=(80, 81), type=")"),
        SemanticToken(range=(81, 82), type="whitespace"),  # " "
        SemanticToken(range=(82, 83), type="{"),
        SemanticToken(range=(83, 85), type="whitespace"),  # "\n\t"
        SemanticToken(range=(85, 86), type="}"),
        SemanticToken(range=(86, 87), type="whitespace"),  # "\n"
        SemanticToken(range=(87, 88), type="}"),
        SemanticToken(range=(88, 89), type="whitespace"),  # "\n"
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
        SemanticToken(range=(0, 8), type="function"),
        SemanticToken(range=(8, 9), type="whitespace"),
        SemanticToken(range=(9, 13), type="identifier"),
        SemanticToken(range=(13, 14), type="("),
        SemanticToken(range=(14, 15), type=")"),
        SemanticToken(range=(15, 16), type="whitespace"),
        SemanticToken(range=(16, 17), type="{"),
        SemanticToken(range=(17, 22), type="whitespace"),
        SemanticToken(range=(22, 36), type="comment"),
        SemanticToken(range=(36, 41), type="whitespace"),
        SemanticToken(range=(41, 44), type="let"),
        SemanticToken(range=(44, 45), type="whitespace"),
        SemanticToken(range=(45, 46), type="identifier"),
        SemanticToken(range=(46, 47), type="whitespace"),
        SemanticToken(range=(47, 48), type="="),
        SemanticToken(range=(48, 49), type="whitespace"),
        SemanticToken(range=(49, 51), type="number"),
        SemanticToken(range=(51, 52), type=";"),
        SemanticToken(range=(52, 53), type="whitespace"),
        SemanticToken(range=(53, 54), type="}"),
        SemanticToken(range=(54, 55), type="whitespace"),
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
        SemanticToken(range=(0, 8), type="#include"),
        SemanticToken(range=(8, 9), type="whitespace"),
        SemanticToken(range=(9, 19), type="system_lib_string"),
        SemanticToken(range=(19, 21), type="\n"),
        SemanticToken(range=(21, 24), type="primitive_type"),
        SemanticToken(range=(24, 25), type="whitespace"),
        SemanticToken(range=(25, 29), type="identifier"),
        SemanticToken(range=(29, 30), type="("),
        SemanticToken(range=(30, 31), type=")"),
        SemanticToken(range=(31, 32), type="whitespace"),
        SemanticToken(range=(32, 33), type="{"),
        SemanticToken(range=(33, 35), type="whitespace"),
        SemanticToken(range=(35, 38), type="namespace_identifier"),
        SemanticToken(range=(38, 40), type="::"),
        SemanticToken(range=(40, 44), type="identifier"),
        SemanticToken(range=(44, 45), type="whitespace"),
        SemanticToken(range=(45, 47), type="<<"),
        SemanticToken(range=(47, 48), type="whitespace"),
        SemanticToken(range=(48, 49), type='"'),
        SemanticToken(range=(49, 61), type="whitespace"),
        SemanticToken(range=(61, 62), type='"'),
        SemanticToken(range=(62, 63), type=";"),
        SemanticToken(range=(63, 65), type="whitespace"),
        SemanticToken(range=(65, 71), type="return"),
        SemanticToken(range=(71, 72), type="whitespace"),
        SemanticToken(range=(72, 73), type="number_literal"),
        SemanticToken(range=(73, 74), type=";"),
        SemanticToken(range=(74, 75), type="whitespace"),
        SemanticToken(range=(75, 76), type="}"),
        SemanticToken(range=(76, 77), type="whitespace"),
    ]


def test_collect_identifiers_go():
    document = Document(
        lang="go",
        content=b'package hello_world\n\n// Hello World\nfunc main() {\na := "Hello World"}\ntype Hello struct {\n\tint field\n}',
    )

    tree = document.parse()

    ids = collect_identifiers(tree, document)

    assert ids == [
        SemanticToken(range=(8, 19), type="package_identifier"),
        SemanticToken(range=(41, 45), type="identifier"),
        SemanticToken(range=(50, 51), type="identifier"),
        SemanticToken(range=(75, 80), type="type_identifier"),
        SemanticToken(range=(91, 94), type="field_identifier"),
        SemanticToken(range=(95, 100), type="type_identifier"),
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
        SemanticToken(range=(5, 8), type="identifier"),
        SemanticToken(range=(16, 19), type="identifier"),
        SemanticToken(range=(25, 29), type="identifier"),
        SemanticToken(range=(34, 37), type="identifier"),
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
        SemanticToken(range=(6, 16), type="identifier"),
        SemanticToken(range=(42, 46), type="identifier"),
        SemanticToken(range=(47, 53), type="type_identifier"),
        SemanticToken(range=(56, 60), type="identifier"),
        SemanticToken(range=(76, 82), type="identifier"),
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
        SemanticToken(range=(6, 11), type="identifier"),
        SemanticToken(range=(25, 29), type="identifier"),
        SemanticToken(range=(42, 43), type="identifier"),
        SemanticToken(range=(46, 53), type="identifier"),
        SemanticToken(range=(54, 57), type="property_identifier"),
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
        SemanticToken(range=(25, 29), type="identifier"),
        SemanticToken(range=(35, 38), type="namespace_identifier"),
        SemanticToken(range=(40, 44), type="identifier"),
        SemanticToken(range=(48, 53), type="identifier"),
        SemanticToken(range=(64, 69), type="type_identifier"),
        SemanticToken(range=(75, 76), type="field_identifier"),
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
