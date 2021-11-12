import csv
import numpy as np

def main():
  with open('sample.csv', mode='r') as csv_file:
    csv_data = csv.reader(csv_file)
    rows = []
    for row in csv_data:
      rows.append([item.strip() for item in row])

    do_iter = True
    iter_count = 0
    if validate_rows(rows):
      while do_iter:
        if iter_count == 0:
          print('Original:\n')
        else:
          print(f'@{iter_count}:\n')
        iter_count += 1
        pretty_print(rows)
        rows, do_iter = simplex_iter(rows)
        print('\n')
      with open('result.csv', 'w') as output:
        output.write(make_csv(rows))

######### MAKE CSV #################
def make_csv(data: list[list]) -> str:
  result = ' '
  for i in data:
    result += ','.join(i) + "\n"
  return result


######### VALIDATION ###############
def validate_rows(rows: list[list[str]]) -> bool:
  if len(rows) < 3:
    print('We should at least have 3 rows')
    return False
  
  rows_len = len(rows[1])
  if len(rows[0]) + 1 != rows_len:
    print('Title row should have 1 element less than other rows.')
    return False
  
  for row in rows[2:]:
    if len(row) != rows_len:
      print('All rows should have the same length')
      return False

  return True


########### PRINTING ################
def pretty_print(matrix: list[list]):
  print('\n'.join([''.join(['{:8}'.format(item) for item in row]) 
      for row in matrix]))


########## SIMPLEX ITER #############
def calculate_steps(rows: np.ndarray, active_c_i):
  result = np.array([])
  for row in rows:
    if row[active_c_i] == 0.0:
      result = np.append(result, -1.0)
    else:
      result = np.append(result, row[-1] / row[active_c_i])
  return result


def find_active_rc(data: list[list[str]]) -> tuple[int, int]:
  num_data_list = [row[1:] for row in data[1:]]
  num_data = np.array(num_data_list)
  num_data = num_data.astype(float)
  zero_row = num_data[0]

  zero_row_sorted = np.sort(zero_row)
  zero_row_sorted = zero_row_sorted[zero_row_sorted < 0.0]
  
  if len(zero_row_sorted) == 0: 
    return (-1, -1)


  for active_zero in zero_row_sorted:
    active_c = int(np.where(zero_row == active_zero)[0])
    steps = calculate_steps(num_data[1:], active_c)
    steps_sorted = np.sort(steps[steps > 0])

    if len(steps_sorted) == 0:
      continue

    active_r = int(np.where(steps == steps_sorted[0])[0]) + 2
    active_c += 1
    return (active_r, active_c)

  return (-1, -1)


def simplex_iter(data: list[list[str]]) -> tuple[list[list[str]], bool]:
  ar, ac = find_active_rc(data)
  if ar == -1 or ac == -1:
    return (data, False)

  num_data_list = [row[1:] for row in data[1:]]
  num_data = np.array(num_data_list)
  num_data = num_data.astype(float)

  anum = num_data[ar - 1][ac - 1]
  ratios = np.array([row[ac - 1] / anum for row in num_data])

  for idx, _ in enumerate(num_data):
    if idx != ar - 1:
      num_data[idx] = num_data[idx] - ratios[idx] * num_data[ar - 1]
  
  ar_ratio = 1 / num_data[ar - 1][ac - 1]
  num_data[ar - 1] = num_data[ar - 1] * ar_ratio

  titles = [row[0] for row in data[1:]]
  titles[ar - 1] = data[0][ac]

  result = num_data.tolist()
  for idx, _ in enumerate(result):
    result[idx] = ['{:.2f}'.format(item) for item in result[idx]]
    result[idx].insert(0, titles[idx])

  result.insert(0, data[0])

  return (result, True)

if __name__ == '__main__':
  main()