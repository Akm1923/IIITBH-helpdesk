import openpyxl

wb = openpyxl.load_workbook(
    r"evals\datasets\IIITBH_QA_Pairs_200.xlsx", read_only=True, data_only=True
)
ws = wb.active
print("sheet", ws.title)
count = 0
for row in ws.iter_rows(values_only=True):
    count += 1
    if count <= 3:
        print("---", count, "ncols", len(row))
        for cell in row:
            text = "" if cell is None else str(cell).replace("\n", " ")
            print(text[:400])
print("rows", count)
