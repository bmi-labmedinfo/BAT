"""
Functions to search for terms specifically in UMLS.

Functions:
    _lookup_sabs(str) -> str
    search_term(str, str, str, str, str) -> List[str]
    get_concepts_for_code(str, List[str], str, str) -> List[str]
    retrieve_names_from_cui(str, List[str], str, str) -> List[str]
"""
from .init import *
import requests


def lookup_sabs(src_lang_code: str):
    """
    Find the MESH sab (i.e. source abbreviation) corresponding to the source language.
    :param src_lang_code: source language code (it,en,fr,es)
    :return: sabs
    """
    sabs_dict = {"it": "MSHITA", "en": "MSH", "es": "MSHSPA", "fr": "MSHFRE"}
    return sabs_dict[src_lang_code]


def get_codes_for_term(apikey: str, text: str, sabs: str, search_type: str = 'exact', version: str = 'current'):
    """
    This function returns the specific sab codes for the given (single) term.
    :param apikey: UMLS personal apikey
    :param text: word for which to find the CUI
    :param sabs: target source abbreviation
    :param search_type: search type
    :param version: version
    :return: List[str]: codes
    """
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/" + version
    full_url = uri + content_endpoint
    page = 0
    code_list = []

    page += 1
    query = {'string': text, 'apiKey': apikey, 'pageNumber': page, 'searchType': search_type,
             'returnIdType': "code", "sabs": sabs}
    r = requests.get(full_url, params=query)
    r.raise_for_status()
    r.encoding = 'utf-8'
    outputs = r.json()
    pages = outputs['pageSize']
    items = (([outputs['result']])[0])['results']

    if len(items) > 0:
        for result in items:
            # print('URI: ' + result['uri'])
            # name = 'Name: ' + result['name']
            # ui_code = 'UI: ' + result['ui']
            code_list.append(result['ui'])
    return code_list


def get_cui_for_code(apikey: str, codes: List[str], sabs: str, version: str = 'current'):
    """
    This function returns UMLS CUIs based on given codes, which must be from the same sabs.
    :param apikey: UMLS personal apikey
    :param codes: all the codes related to a single term of the specified sabs.
    :param sabs: source abbreviation of the codes.
    :param version: version
    :return: List[str]: UMLS CUIs
    """
    base_uri = 'https://uts-ws.nlm.nih.gov'
    cui_list = []

    for code in codes:
                path = '/search/' + version
                query = {'apiKey': apikey, 'string': code, 'rootSource': sabs, 'inputType': 'sourceUI', 'pageNumber': 1,
                         'tty': 'SY'}
                output = requests.get(base_uri + path, params=query)
                items = output.json()


                output.encoding = 'utf-8'
                # print(output.url)

                output_json = output.json()
                results = (([output_json['result']])[0])['results']

                if len(results) > 0:

                    for item in results:
                        # print('CUI: ' + item['ui'] + '\n' + 'Name: ' + item['name'] + '\n')
                        # cui = 'CUI: ' + item['ui']
                        # name = 'Name: ' + item['name']
                        cui_list.append(item['ui'])

    return cui_list


def retrieve_names_from_cui(apikey: str, codes: List[str], sabs: str, version: str = 'current'):
    """
    This function returns atoms (names) from CUI input.
    :param apikey: UMLS personal apikey
    :param codes: UMLS CUIs
    :param sabs: final target source abbreviation
    :param version: version
    :return: atoms (names)
    """
    uri = 'https://uts-ws.nlm.nih.gov'
    names = []

    for code in codes:
        content_endpoint = '/rest/content/' + str(version) + '/CUI/' + str(code)

        query = {'apiKey': apikey}
        r = requests.get(uri + content_endpoint, params=query)
        r.encoding = 'utf-8'

        if r.status_code != 200:
            continue

        items = r.json()
        json_data = items['result']
        atoms = json_data['atoms']

        pages = items['pageSize']
        for page in np.arange(pages):
            try:
                #page += 1
                atom_query = {'apiKey': apikey, 'pageNumber': page}
                a = requests.get(atoms, params=atom_query)
                a.encoding = 'utf-8'

                if a.status_code != 200:
                    continue
                all_atoms = a.json()
                json_atoms = all_atoms['result']
                for atom in json_atoms:
                    if atom['rootSource'] == sabs:
                        # name= 'Name: ' + atom['name']
                        # cui='CUI: ' + jsonData['ui']
                        # aui='AUI: ' + atom['ui']
                        # term_type='Term Type: ' + atom['termType']
                        # code='Code: ' + atom['code']
                        # source='Source Vocabulary: ' + atom['rootSource']
                        names.append(atom['name'])
            except: ""
    return names
