import csv

# 输入和输出文件的路径
input_csv_path = '/Users/lifushen/Desktop/giana1/data_clean.csv'
output_csv_path = '/Users/lifushen/Desktop/giana1/output_data.csv'

# 打开输入文件进行读取，打开输出文件准备写入，这次指定制表符为分隔符
with open(input_csv_path, newline='') as infile, open(output_csv_path, 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter='\t')  # 指定制表符为分隔符
    writer = csv.writer(outfile, delimiter='\t')  # 保持输出文件也使用制表符分隔

    # 读取第一行（标题行）
    headers = next(reader)

    # 现在使用正确的列名和分隔符
    cdr3_index = headers.index('cdr3')
    v_segm_index = headers.index('v.segm')

    # 将`cdr3`和`v.segm`放在标题行的最前面
    reordered_headers = [headers[cdr3_index], headers[v_segm_index]] + [h for i, h in enumerate(headers) if
                                                                        i not in [cdr3_index, v_segm_index]]
    writer.writerow(reordered_headers)

    # 遍历文件的剩余行，并按照新的列顺序重新排列每一行的数据
    for row in reader:
        reordered_row = [row[cdr3_index], row[v_segm_index]] + [v for i, v in enumerate(row) if
                                                                i not in [cdr3_index, v_segm_index]]
        writer.writerow(reordered_row)
