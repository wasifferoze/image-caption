import string

# load doc files
def load_doc(file_name):
    file = open(file_name, 'r')
    txt = file.read()
    file.close()
    return txt


# images description extraction
def load_desc(desc_doc):
    mapp_dict = dict()
    for line in desc_doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        img_id, img_desc = tokens[0], tokens[1:]
        img_id = img_id.split('.')[0]
        img_desc = ' '.join(img_desc)
        if img_id not in mapp_dict:
            mapp_dict[img_id] = img_desc
    return mapp_dict


# for cleaning description text
def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = string.maketrans('', '')
    for key, desc in descriptions.items():
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table, string.punctuation) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word) > 1]
        # store as string
        descriptions[key] = ' '.join(desc)


# save descriptions to file, one per line
def save_doc(descriptions, filename):
    lines = list()
    for key, desc in descriptions.items():
        lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# to use this code
file_name = 'data/Flickr8k_text/Flickr8k.token.txt'
# load file
doc = load_doc(file_name)
# parse description
descriptions = load_desc(doc)
print ('Loaded: %d' % len(descriptions))
# clean here
clean_descriptions(descriptions)
# vocabulary summary
all_tokens = ' '.join(descriptions.values()).split()
vocab = set(all_tokens)
print ('Vocabulary Size: %d' % len(vocab))
# save doc
save_doc(descriptions, 'captions.txt')