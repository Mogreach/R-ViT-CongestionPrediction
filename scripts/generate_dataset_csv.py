import csv
import os
def main(file_path,dataset_type):
    file_name_list=os.listdir(file_path)
    length = len(file_name_list)
    train_csv = open(f"./files/{dataset_type}_train.csv",'w',newline='')
    test_csv  = open(f"./files/{dataset_type}_test.csv",'w',newline='')
    train_writer = csv.writer(train_csv)
    test_writer  = csv.writer(test_csv)
    for i,each in enumerate(file_name_list):
        row = [f"feature/{each}", f"label/{each}"]
        if i+1<=(length*8/10):
            train_writer.writerow(row)
        else:
            test_writer.writerow(row)
    train_csv.close()
    test_csv.close()

def generate_validation_csv(test_csv):
    file = open("./files/validation.csv",'w',newline='')
    validation_csv=csv.writer(file)
    with open(test_csv,'r') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i < 300:
                validation_csv.writerow(row)
    file.close()

main("../CircuitNet-28/congestion/feature","CircuitNet-28")
# generate_validation_csv("./files/CircuitNet-28_test.csv")