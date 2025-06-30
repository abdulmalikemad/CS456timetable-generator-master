import json
import random
from costs import check_hard_constraints, subjects_order_cost, empty_space_groups_cost, empty_space_teachers_cost, \
    free_hour
from model import Class, Classroom, Data


import random
from model import Class, Classroom, Data

def load_txt_data(file_path, teachers_empty_space, groups_empty_space, subjects_order):
    """
    Loads input data from a text file and prepares all required structures for scheduling.

    :param file_path: path to the input .txt file
    :param teachers_empty_space: dict to store empty rows per teacher
    :param groups_empty_space: dict to store empty rows per group
    :param subjects_order: dict to track order of (T, P, V) classes per subject/group
    :return: Data object with groups, teachers, classes, and classrooms
    """
    classes = {}
    classrooms = {}
    teachers = {}
    groups = {}
    class_list = []
    classroom_types = {}

    def classify_room(room_name):
        if "معمل" in room_name:
            return "P"  # practical
        elif "مدرج" in room_name:
            return "T"  # theoretical
        else:
            return "X"  # unknown/other

    with open(file_path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            parts = [x.strip().strip(',') for x in line.strip().split(",") if x.strip().strip(',') != '']

            if len(parts) < 7:
                continue  # skip invalid lines

            subject, instructor, typ, duration, allowed_rooms, group, group_size = parts
            duration = int(duration)
            group_size = int(group_size)
            allowed_room_list = allowed_rooms.split("|")

            # Initialize teacher empty space
            if instructor not in teachers_empty_space:
                teachers_empty_space[instructor] = []
            if instructor not in teachers:
                teachers[instructor] = len(teachers)

            # Initialize group
            if group not in groups:
                groups[group] = len(groups)
                groups_empty_space[groups[group]] = []

            # Init subject order
            key = (subject, groups[group])
            if key not in subjects_order:
                subjects_order[key] = [-1, -1, -1]  # [T, V, P] or whatever order you use

            # Init allowed classrooms
            for room in allowed_room_list:
                if room not in classroom_types:
                    classroom_types[room] = classify_room(room)

            # Create the class
            new_class = Class([group], instructor, subject, typ, duration, allowed_room_list)
            class_list.append(new_class)

    # Shuffle to spread load (especially teachers)
    random.shuffle(class_list)

    # Assign class IDs
    for cl in class_list:
        classes[len(classes)] = cl

    # Convert classroom names to Classroom objects
    for room_name, room_type in classroom_types.items():
        classrooms[len(classrooms)] = Classroom(room_name, room_type)

    # Convert class.allowed_rooms from names to indices
    for i, cl in classes.items():
        cl.classrooms = [idx for idx, c in classrooms.items() if c.name in cl.classrooms]
        cl.groups = [groups[g] for g in cl.groups]

    return Data(groups, teachers, classes, classrooms)



def set_up(num_of_columns):
    """
    Sets up the timetable matrix and dictionary that stores free fields from matrix.
    :param num_of_columns: number of classrooms
    :return: matrix, free
    """
    w, h = num_of_columns, 54  # 6 أيام * 9 ساعات

    matrix = [[None for x in range(w)] for y in range(h)]
    free = []

    # initialise free dict as all the fields from matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            free.append((i, j))
    return matrix, free


def generate_html_timetable(matrix, classrooms, data, output_file='timetable.html'):
    days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
    hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # 9 ساعات
    rows_per_day = len(hours)

    # عكس الفهرسة للوصول لاسم المادة
    reverse_classes = {v: k for k, v in data.classes.items()}

    html = """
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; text-align: center; font-family: Arial, sans-serif; }
            th, td { border: 1px solid #333; padding: 10px; vertical-align: top; }
            th { background-color: #eee; }
        </style>
    </head>
    <body>
    <h2>Timetable</h2>
    <table>
        <tr>
            <th>Day / Time</th>
    """

    for hour in hours:
        html += f"<th>{hour}:00</th>"
    html += "</tr>\n"

    for day_index in range(6):  # Adjusted to Saturday–Thursday
        html += f"<tr><td><b>{days[day_index]}</b></td>"
        for hour_index in range(len(hours)):
            row_index = day_index * rows_per_day + hour_index
            cell_data = ""
            for col_index in range(len(matrix[0])):
                class_id = matrix[row_index][col_index]
                if class_id is not None:
                    cls = data.classes[class_id]
                    room_name = data.classrooms[col_index].name
                    cell_data += f"{cls.subject}<br>{cls.teacher}<br>{room_name}<hr>"
            html += f"<td>{cell_data.strip('<hr>')}</td>"
        html += "</tr>\n"

    html += """
    </table>
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ HTML جدول الحصص تم إنشاؤه: {output_file}")



def write_solution_to_file(matrix, data, filled, filepath, groups_empty_space, teachers_empty_space, subjects_order):
    """
    Writes statistics and schedule to file.
    """
    f = open('solution_files/sol_' + filepath, 'w')

    f.write('-------------------------- STATISTICS --------------------------\n')
    cost_hard = check_hard_constraints(matrix, data)
    if cost_hard == 0:
        f.write('\nHard constraints satisfied: 100.00 %\n')
    else:
        f.write('Hard constraints NOT satisfied, cost: {}\n'.format(cost_hard))
    f.write('Soft constraints satisfied: {:.02f} %\n\n'.format(subjects_order_cost(subjects_order)))

    empty_groups, max_empty_group, average_empty_groups = empty_space_groups_cost(groups_empty_space)
    f.write('TOTAL empty space for all GROUPS and all days: {}\n'.format(empty_groups))
    f.write('MAX empty space for GROUP in day: {}\n'.format(max_empty_group))
    f.write('AVERAGE empty space for GROUPS per week: {:.02f}\n\n'.format(average_empty_groups))

    empty_teachers, max_empty_teacher, average_empty_teachers = empty_space_teachers_cost(teachers_empty_space)
    f.write('TOTAL empty space for all TEACHERS and all days: {}\n'.format(empty_teachers))
    f.write('MAX empty space for TEACHER in day: {}\n'.format(max_empty_teacher))
    f.write('AVERAGE empty space for TEACHERS per week: {:.02f}\n\n'.format(average_empty_teachers))

    f_hour = free_hour(matrix)
    if f_hour != -1:
        f.write('Free term -> {}\n'.format(f_hour))
    else:
        f.write('NO hours without classes.\n')

    groups_dict = {}
    for group_name, group_index in data.groups.items():
        if group_index not in groups_dict:
            groups_dict[group_index] = group_name
    days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
    hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]


    f.write('\n--------------------------- SCHEDULE ---------------------------')
    for class_index, times in filled.items():
        c = data.classes[class_index]
        groups = ' '
        for g in c.groups:
            groups += groups_dict[g] + ', '
        f.write('\n\nClass {}\n'.format(class_index))
        f.write('Teacher: {} \nSubject: {} \nGroups:{} \nType: {} \nDuration: {} hour(s)'
                .format(c.teacher, c.subject, groups[:len(groups) - 2], c.type, c.duration))
        room = str(data.classrooms[times[0][1]])
        f.write('\nClassroom: {:2s}\nTime: {}'.format(room[:room.rfind('-')], days[times[0][0] // 9]))
        for time in times:
            f.write(' {}'.format(hours[time[0] % 9]))

    f.close()


def show_statistics(matrix, data, subjects_order, groups_empty_space, teachers_empty_space):
    """
    Prints statistics.
    """
    cost_hard = check_hard_constraints(matrix, data)
    if cost_hard == 0:
        print('Hard constraints satisfied: 100.00 %')
    else:
        print('Hard constraints NOT satisfied, cost: {}'.format(cost_hard))
    print('Soft constraints satisfied: {:.02f} %\n'.format(subjects_order_cost(subjects_order)))

    empty_groups, max_empty_group, average_empty_groups = empty_space_groups_cost(groups_empty_space)
    print('TOTAL empty space for all GROUPS and all days: ', empty_groups)
    print('MAX empty space for GROUP in day: ', max_empty_group)
    print('AVERAGE empty space for GROUPS per week: {:.02f}\n'.format(average_empty_groups))

    empty_teachers, max_empty_teacher, average_empty_teachers = empty_space_teachers_cost(teachers_empty_space)
    print('TOTAL empty space for all TEACHERS and all days: ', empty_teachers)
    print('MAX empty space for TEACHER in day: ', max_empty_teacher)
    print('AVERAGE empty space for TEACHERS per week: {:.02f}\n'.format(average_empty_teachers))

    f_hour = free_hour(matrix)
    if f_hour != -1:
        print('Free term ->', f_hour)
    else:
        print('NO hours without classes.')
