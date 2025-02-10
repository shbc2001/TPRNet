class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset=='arcade':
            return './data/Arcade/'
        elif dataset=='stenosis':
            return './data/stenosis/'
        elif dataset=='fives':
            return './data/FIVES/'
        else:
            print("no such dataset "+dataset)
