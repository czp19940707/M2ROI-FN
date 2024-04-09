import os

if __name__ == '__main__':
    # ['v1', 'v2', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15']
    # version_list = ['v1', 'v2', 'v3', 'v4', 'v5', 'z6', 'z7', 'z8', 'z9', 'z10']
    version_list = ['v1', 'v2', 'v3', 'v4', 'v5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15']
    for version in version_list:
        os.system(r'python eval.py -v {}'.format(version))