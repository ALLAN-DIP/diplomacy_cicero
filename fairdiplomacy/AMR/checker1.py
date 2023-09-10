# of the form XXX (arrangement)
TOKEN_SINGLE = [
    "PRP",
    "XDO",
    "REJ",
    "YES",
    "HUH",
    "NOT",
    "NAR",
    "BWX",
    "FCT",
    "INS",
    "QRY",
    "THK",
    "IDK",
    "SUG",
    "HOW",
    "WHT",
    "SRY",
    "WHY",
    "POB"
]

# of the form XXX (power power ...) or XXX (arrangement) (arrangement) ... or similar
TOKEN_MULTIPLE = [
    "PCE",
    "AND",
    "ORR",
    "SCD",
    "DMZ",
    "OCC"
] 
# TODO: EXP, FOR, IFF, XOY, YDO, SND, FWD, BCC

# children of these nodes can appear in any order
NEEDS_SORTING = ["ALY", "VSS", "AND", "ORR", "PCE", "SCD", "OCC"]

POWERS = ("FRA", "GER", "AUS", "RUS", "ENG", "ITA", "TUR")

PROVINCES = ("ALB",
    "ANK",
    "APU",
    "ARM",
    "BEL",
    "BER",
    "BRE",
    "BUL",
    "CLY",
    "CON",
    "DEN",
    "EDI",
    "FIN",
    "GAS",
    "GRE",
    "HOL",
    "KIE",
    "LON",
    "LVN",
    "LVP",
    "MAR",
    "NAF",
    "NAP",
    "NWY",
    "PIC",
    "PIE",
    "POR",
    "PRU",
    "ROM",
    "RUM",
    "SEV",
    "SMY",
    "SPA",
    "STP",
    "SWE",
    "SYR",
    "TRI",
    "TUN",
    "TUS",
    "VEN",
    "YOR",
    "WAL",
    "BOH",
    "BUD",
    "BUR",
    "MOS",
    "MUN",
    "GAL",
    "PAR",
    "RUH",
    "SER",
    "SIL",
    "TYR",
    "UKR",
    "VIE",
    "WAR",
    "ADR",
    "AEG",
    "BAL",
    "BAR",
    "BLA",
    "BOT",
    "EAS",
    "ECH",
    "HEL",
    "ION",
    "IRI",
    "LYO",
    "MAO",
    "NAO",
    "NTH",
    "NWG",
    "SKA",
    "TYS",
    "WES"
)

class Node:
    def __init__(self, name: str):
        """
        name:           str, the name of the node, generally a three-letter DAIDE token
        subtree_name:   str, a string representation of the entire node subtree
        children:       list, a list of the node's children
        parent:         Node, the parent of the current node
        """
        self.name = name
        self.subtree_name = name
        self.children = []
        self.parent = None
        self.token = name

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.name

def create_tree(daide: str) -> Node:
    """
    Creates a tree from the given DAIDE string
    
    Parameter:
        daide:  str, the DAIDE string to turn into a tree
    Returns:
        root:   Node, the root of the tree
    """
    if daide == None or daide == "":
        return Node("root")

    # so that the DAIDE string can be split on whitespace, preserving tokens and parentheses    
    daide = daide.replace("(", " ( ")
    daide = daide.replace(")", " ) ")
    daide_tokens = daide.split()
    daide_tokens.insert(0, "(")
    daide_tokens.append(")")

    if daide_tokens.count(")") > daide_tokens.count("("):
        daide_tokens = ["("] * (daide_tokens.count(")") - daide_tokens.count("(")) + daide_tokens

    # do it recursively
    root = Node("root")
    parent = root
    current = root

    for t in daide_tokens:
        if t == "(":
            # start a new node and append to the parent
            current = Node("(")
            current.parent = parent
            parent.children.append(current)
            parent = current

        elif t == ")":
            # return to the parent
            parent = parent.parent
        else:
            current = Node(t)
            current.parent = parent
            parent.children.append(current)
    
    _restructure(root)
    _rename_subtree(root)
    _sort_tree(root)
    _rename_name(root)

    return root

def _restructure(root: Node):
    """
    Condenses the DAIDE tree by removing auxilliary and unnecessary nodes.
    Restructures the tree where necessary.
    """

    # if the keyword is of form similar to PRP (arrangement), then the children are
    # represented as [PRP, arrangement]
    # rearrange the nodes such that arrangement's children are the children of PRP
    # i.e. instead of:                           we have:
    #           node                                node
    #           /  \                                  |          
    #         PRP  XDO-RUS-AMY...                    PRP
    #                \                                |
    #                XDO                             XDO
    #
    if len(root.children) == 2:
        first_child = root.children[0]
        if first_child.name in TOKEN_SINGLE:
            children = root.children[1].children
            first_child.children = children
            root.children.remove(root.children[1])
            for node in first_child.children:
                node.parent = first_child
    
    # rearrange such that the second child onwards is the child of the first child
    # i.e. instead of:                           we have:
    #             node                                node
    #         /    |    \                               |          
    #       AND   XDO   XDO                            AND
    #                                                 /   \
    #                                               XDO   XDO
    #
    if len(root.children) > 1 and root.children[0].name in TOKEN_MULTIPLE:
        first_child = root.children[0]
        while len(root.children) > 1:
            root.children[1].parent = first_child
            first_child.children.append(root.children[1])
            root.children.remove(root.children[1])

    # remove extra unncessary node between nodes
    #       node             becomes        node
    #        |                               |
    #     PRP-XDO-...                       PRP
    #        |
    #       PRP
    #
    if root.name == "(":
        parent = root.parent
        if len(root.children) == 1:
            parent.children[parent.children.index(root)] = root.children[0]
            root.children[0].parent = parent

    # specific case for SCD
    if root.parent != None and root.parent.name == "SCD" and len(root.children) > 0:
        # first child should be the power, second child onwards should be SC
        root.name = root.children[0].name
        root.children.remove(root.children[0])

    # specific case for PCE
    if root.name == "PCE" and len(root.children) == 1:
        root.children = root.children[0].children
        for node in root.children:
            node.parent = root

    # rearrange ALY-VSS to make it easier to compare equality
    #                    node
    #     ALY    FRA-ENG     VSS     TUR-RUS
    #           FRA   ENG           TUR    RUS
    #
    # to
    #                   node
    #               ALY      VSS
    #            FRA ENG   TUR RUS
    #
    # shouldn't fail now if missing VSS
    if len(root.children) > 0 and root.children[0].name == "ALY":
        if len(root.children) > 1:
            aly = root.children[0]
            aly.children = root.children[1].children
            for child in aly.children:
                child.parent = aly
        
            if len(root.children) > 3 and root.children[2].name == "VSS":
                vss = root.children[2]
                vss.children = root.children[3].children
                for child in vss.children:
                    child.parent = vss
                
                root.children.remove(root.children[3])
        root.children.remove(root.children[1])

    for node in root.children:
        _restructure(node)

def _rename_subtree(root: Node):
    """
    Rename subtrees in the tree from the bottom up, to aid in sorting the tree later
    Slight issues with renaming for ALY-VSS nodes
    """
    for node in root.children:
        _rename_subtree(node)
    
    if len(root.children) > 0:
        if root.name == "(":
            root.subtree_name = "-".join([n.subtree_name for n in root.children])
        else:
            root.subtree_name = root.name + "-" + "-".join([n.subtree_name for n in root.children])

def _rename_name(root: Node):
    """
    Rename nodes in the tree from the bottom up, to aid in printing
    Slight issues with renaming for ALY-VSS nodes
    """
    for node in root.children:
        _rename_name(node)
    
    if len(root.children) > 0:
        if root.name == "(":
            root.name = "-".join([n.name for n in root.children])

def _sort_tree(root: Node):
    """
    Sort the nodes of the tree that are order-agnostic, e.g. children of AND, VSS, ...
    """
    if root.name in NEEDS_SORTING:
        root.children.sort(key=lambda x: x.subtree_name)

    # implementation specific for SCD, DMZ: must sort their children's children
    if root.name == "SCD" or root.name == "DMZ":
        for child in root.children:
            child.children.sort(key=lambda x: x.subtree_name)
    
    for node in root.children:
        _sort_tree(node)

def visit(root: Node):
    """
    Basic visit function that prints out the node name and its children
    """
    print(f"node: {root}, children: {root.children}")
    for node in root.children:
        visit(node)

def tree_equal(a: Node, b: Node) -> bool:
    """
    Given the root nodes of two DAIDE trees, checks if the two trees are the same
    """
    if a == None or b == None:
        return False
    
    if a.name != b.name:
        return False

    if len(a.children) != len(b.children):
        return False
    
    # recursively check if the children are equal
    equal = True
    for a1, b1 in zip(a.children, b.children):
        equal = equal and tree_equal(a1, b1)
    
    return equal

def daide_equal(a: str, b: str) -> bool:
    """
    Checks if two DAIDE strings are equal by converting into a tree first
    """
    a_tree = create_tree(a)
    b_tree = create_tree(b)

    return tree_equal(a_tree, b_tree)

def _calculate_similarity(a: Node, b: Node) -> tuple:
    """
    Determine the similarity between tree a and tree b.
    Returns:
        match:  int, represents how many nodes in `a` matched the corresponding node in `b`
        total:  int, represents the total number of nodes checked
    """
    match = 0
    total = 0

    if a.name == b.name and len(a.children) == len(b.children):
        match += 1
    total += 1
    
    for a1, b1 in zip(a.children, b.children):
        m, t = _calculate_similarity(a1, b1)
        match += m
        total += t
    
    return match, total

def get_accuracy(a: str, b: str) -> float:
    """
    Get the similarity between DAIDE string a and DAIDE string b.
    Returns a number between 0 and 1, where 1 represents strings are equal.
    """
    match, total = _calculate_similarity(create_tree(a), create_tree(b))
    return match / total


def _get_subtrees(root: Node) -> list:
    """
    Get a list of all subtrees as a string representation
    of the form root-children
    """
    subtrees = []
    if len(root.children) > 0:
        subtree_name = root.name + "-" + "-".join(c.name for c in root.children)
    else:
        subtree_name = root.name
    subtrees.append(subtree_name)
    
    for child in root.children:
        subtrees += _get_subtrees(child)
    
    return subtrees

def _replace(root: Node):
    """
    Replaces all power and province names within a tree with POWER and PROVINCE respectively.
    """
    for child in root.children:
        _replace(child)
    
    if root.name in POWERS:
        root.name = "POWER"
    
    elif root.name in PROVINCES:
        root.name = "PROVINCE"
    
    elif root.token == "(":
        root.name = "-".join(c.name for c in root.children)

def f_score(a: str, b: str, replace_name=False) -> float:
    """
    Calculates a more accrate F-score by calculating the overlap between parent-children pairs
    of two DAIDE strings.

    Parameter:
        a:              str, the first DAIDE string
        b:              str, the second DAIDE string
        replace_name:   bool, whether province and power names should be replaced to compare only structure
    Returns:
        f_score:        float, the f-score of parent-children pairs
    """
    tree_a = create_tree(a)
    tree_b = create_tree(b)

    if replace_name:
        _replace(tree_a)
        _replace(tree_b)

    subtrees_a = _get_subtrees(tree_a)
    subtrees_b = _get_subtrees(tree_b)

    tp = 0
    for s in subtrees_a:
        if s in subtrees_b:
            tp += 1

    # fp = selected and not correct
    fp = len(subtrees_a) - tp

    # fn = not selected and correct
    fn = len(subtrees_b) - tp

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)

    if (precision + recall) == 0:
        return 0
        
    f_score = (2 * precision * recall) / (precision + recall)

    return f_score