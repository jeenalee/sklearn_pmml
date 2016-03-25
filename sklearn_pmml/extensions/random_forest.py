from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import tree
from sklearn.tree._tree import Tree
from xml.etree import ElementTree
from bs4 import BeautifulSoup
from collections import deque
import numpy

def from_pmml(self, pmml):
    """Returns a random forest model with trees extracted from the PMML file."""

    model = self()
    
    # Create a dictionary to hold the URI of PMML.
    uri = {'pmml': 'http://www.dmg.org/PMML-4_2'}
    
    # Parse PMML to xml tree and soup.
    #    xml tree: used for tree traversal.
    #    soup: used for extracting information that is
    #          scattered in the PMML file.
    pmml_root = _parse_pmml_to_xml(pmml)
    pmml_soup = _parse_pmml_to_soup(pmml)

    # Create a list of trees in xml representation.
    list_trees = _find_trees(pmml_root, uri)
    
    # Number of trees in RandomForestClassifier == len(list_trees)
    model.n_estimators = len(list_trees)
    
    # Create a list to hold the trees.
    model.estimators_ = []

    # Information required for initializing DecisionTreeClassifier.
    n_features, n_classes, n_outputs = _extract_tree_info(pmml_soup)

    # Adding trees to the RandomForestClassifier.
    for i in range(0, model.n_estimators):
        current_tree = list_trees[i]
        
        # Create a DecisionTreeClassifier
        model_tree = _initiate_tree(n_features, n_classes, n_outputs)
        _fill_in_tree(model_tree, current_tree, uri)

        # TODO: numpy.append does not work (object not writable)
        #    AttributeError: attribute 'threshold' of
        #    'sklearn.tree._tree.Tree' objects is not writable Add the
        # DecisionTreeClassifier to RandomForestClassifier.
        model.estimators_.append(model_tree)
        
    return model


def _fill_in_tree(model_tree, xml_tree, uri_d):
    """Fill in model_tree given a xml (PMML) representation of a tree."""

    # Find the beginning of the tree. It starts with <Node id="1">.
    pmml_root = xml_tree.find("pmml:TreeModel", uri_d).find("pmml:Node", uri_d)

    dict_t = {}
    dict_t['children_left'] = []
    dict_t['children_right'] = []
    dict_t['threshold'] = []
    dict_t['feature'] = []
    
    _walk_tree(pmml_root, _extract_info_from_node, uri_d, dict_t)
    
    for key in dict_t:
         value = dict_t[key]
         dict_t[key] = numpy.array(value)
   
        
def _walk_tree(t, fn, uri_d, dict_t):
    """Walks down the tree in breadth-first order, and performs fn on each node."""
    queue = deque()
    queue.append(t)

    while len(queue) != 0:
        current_t = queue.popleft()
        _visit_node(current_t, fn, queue, dict_t, uri_d)

        
def _visit_node(node, fn, l, d, uri_d):
    """ 
    Visit a node, apply the function fn, and add queue to list l. 
    Add values to the dictionary d, and returns it.
    """
    left = None
    right = None
    
    for child in node:
        if child.tag != "{http://www.dmg.org/PMML-4_2}Node":
            continue
        elif left is None:
            left = child
        elif right is None:
            right = child
        else:
            # If left and right are not None and the function
            # encounters another child, the given tree is not a binary
            # tree.
            raise Exception("This is not a binary tree.")

    fn(node, left, right, d, uri_d)

    if left is not None:
        l.append(left)
    if right is not None:
        l.append(right)
    
    return d


def _extract_info_from_node(node, left, right, d, uri_d):
    # When extracting left children, also extracts threshold and feature.
    _extract_children_left(left, d, uri_d)
    _extract_children_right(right, d)

    return d


def _extract_children_left(left, d, uri_d):
    pred = None
    if left is not None:
        left_id = int(left.attrib["id"]) - 1
        pred = left.find("pmml:SimplePredicate", uri_d)
    else:
        left_id = -1
    d['children_left'].append(left_id)
    
    _extract_threshold(pred, d)
    _extract_feature(pred, d)
    
    return d


def _extract_children_right(right, d):
    if right is not None:
        right_id = int(right.attrib["id"]) - 1
    else:
        right_id = -1
    d['children_right'].append(right_id)
     
    return d


def _extract_threshold(pred, d):
    if pred is not None:
        threshold = pred.attrib["value"]
    else:
        threshold = -2
    d['threshold'].append(threshold)
    
    return d


def _extract_feature(pred, d):
    if pred is not None:
        x_feature = pred.attrib["field"]
        # Remove x, and transform it into an integer.
        feature = int(x_feature[1:])
    else:
        feature = -2
    d['feature'].append(feature)
    
    return d


def _initiate_tree(n_features, n_classes, n_outputs):
    """Returns a DecisionTreeClassifier with the appropriate parameters."""
    model_tree = tree.DecisionTreeClassifier()
    model_tree.tree_ = Tree(n_features, n_classes, n_outputs)

    return model_tree


def _find_trees(xml_root, uri_d):
    """Returns a list of trees in xml representation."""
    miningmodel = xml_root.find('pmml:MiningModel', uri_d)
    segmentation = miningmodel.find('pmml:Segmentation', uri_d)
    # 'trees' is a list of trees.
    trees = segmentation.findall('pmml:Segment', uri_d)

    return trees


def _parse_pmml_to_xml(pmml):
    """Returns a Python xml object for a given pmml."""
    xml = ElementTree.parse(pmml)
    xml_root = xml.getroot()

    return xml_root


def _parse_pmml_to_soup(pmml):
    """Parses PMML to a soup with BeautifulSoup."""
    with open(pmml, 'r') as f:
        soup = BeautifulSoup(f, 'xml')

    return soup

                       
def _extract_tree_info(soup):
    n_classes = len(soup.find_all('Value'))
    # Transforming it into numpy array to match the sklearn internal
    # Tree format.
    n_classes = numpy.array([n_classes])
    # Number of features is length of DataField - 1.
    # 1 is for the predicted value.
    n_features = len(soup.find_all('DataField')) - 1
    n_outputs = 1
    return n_features, n_classes, n_outputs


RandomForestClassifier.from_pmml = classmethod(from_pmml)
