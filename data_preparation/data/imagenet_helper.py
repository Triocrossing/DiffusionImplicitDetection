import json
import os
class class_index_imagenet:
    def __init__(self, file_name="data/imagenet_class_index.json"):
        # assert file exist
        assert os.path.exists(file_name), f"File {file_name} does not exist"
        self.class_index = json.load(open(file_name, "r"))
    
    def get_class_name_from_id(self, class_id, remove_underline=True):
        if remove_underline:
            return self.class_index[str(class_id)][1].replace("_", " ")
        else:
            return self.class_index[str(class_id)][1]
          
    def get_class_name_from_train_prefix(self, prefix, remove_underline=True):
        class_name = next((v[1] for k, v in self.class_index.items() if v[0] == prefix), None)
        return class_name.replace("_", " ") if remove_underline and class_name else class_name

    # basefile namae can be :n02281787_4560.JPEG
    # or 428_xxx_00113.png
    def auto_filename_parser(self, filename):
        if filename.startswith("n"):
            return self.get_class_name_from_train_prefix(filename.split("_")[0])
        # if it is a number
        elif filename[0].isdigit():
            return self.get_class_name_from_id(int(filename.split("_")[0]))
        else:
            return None

def unit_test_class_index_imagenet():
    file_name = "data/imagenet_class_index.json"

    class_index_imagenet_obj = class_index_imagenet(file_name)
    assert class_index_imagenet_obj.get_class_name_from_id(0) == "tench", "class_index_imagenet_obj.get_class_name_from_id(0) failed"
    assert class_index_imagenet_obj.get_class_name_from_train_prefix("n01440764") == "tench", "class_index_imagenet_obj.get_class_name_from_train_prefix(n01440764) failed"
    
    assert class_index_imagenet_obj.get_class_name_from_id(1) == "goldfish", "class_index_imagenet_obj.get_class_name_from_id(1) failed" 
    assert class_index_imagenet_obj.get_class_name_from_train_prefix("n01443537") == "goldfish", "class_index_imagenet_obj.get_class_name_from_train_prefix(n01443537) failed"
    
    assert class_index_imagenet_obj.get_class_name_from_id(765) == "rocking chair", "class_index_imagenet_obj.get_class_name_from_id(765) failed"
    assert class_index_imagenet_obj.get_class_name_from_train_prefix("n04099969") == "rocking chair", "class_index_imagenet_obj.get_class_name_from_train_prefix(n04099969) failed"
    
    
    # test auto_filename_parser
    filenames = ["n02281787_4560.JPEG", "428_xxx_00113.png"]
    assert class_index_imagenet_obj.auto_filename_parser(filenames[0]) == "lycaenid", "class_index_imagenet_obj.auto_filename_parser(n02281787_4560.JPEG) failed"
    assert class_index_imagenet_obj.auto_filename_parser(filenames[1]) == "barrow", "class_index_imagenet_obj.auto_filename_parser(428_xxx_00113.png) failed"
    
    print("All tests passed")
    
def main():
    unit_test_class_index_imagenet()
    
# main
if __name__ == "__main__":
    main()