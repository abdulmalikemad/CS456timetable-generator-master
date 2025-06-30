def subjects_order_cost(subjects_order):
    cost = 0
    total = 0

    for (subject, group_index), times in subjects_order.items():
        if times[0] != -1 and times[1] != -1:
            total += 1
            if times[0] > times[1]:
                cost += 1

        if times[0] != -1 and times[2] != -1:
            total += 1
            if times[0] > times[2]:
                cost += 1

        if times[1] != -1 and times[2] != -1:
            total += 1
            if times[1] > times[2]:
                cost += 1

    if total == 0:
        return 100  # ما فيش ولا حالة تحتاج تقييم → اعتبرها 100% صح

    return 100 * (total - cost) / total



def empty_space_groups_cost(groups_empty_space):
    cost = 0
    max_empty = 0

    for group_index, times in groups_empty_space.items():
        times.sort()
        empty_per_day = {i: 0 for i in range(6)}  # 6 أيام

        for i in range(1, len(times) - 1):
            a = times[i-1]
            b = times[i]
            diff = b - a
            if a // 9 == b // 9 and diff > 1:
                empty_per_day[a // 9] += diff - 1
                cost += diff - 1

        for value in empty_per_day.values():
            if max_empty < value:
                max_empty = value

    return cost, max_empty, cost / len(groups_empty_space)



def empty_space_teachers_cost(teachers_empty_space):
    cost = 0
    max_empty = 0

    for teacher_name, times in teachers_empty_space.items():
        times.sort()
        empty_per_day = {i: 0 for i in range(6)}  # 6 أيام

        for i in range(1, len(times) - 1):
            a = times[i - 1]
            b = times[i]
            diff = b - a
            if a // 9 == b // 9 and diff > 1:
                empty_per_day[a // 9] += diff - 1
                cost += diff - 1

        for value in empty_per_day.values():
            if max_empty < value:
                max_empty = value

    return cost, max_empty, cost / len(teachers_empty_space)


def free_hour(matrix):
    days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
    hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]

    for i in range(len(matrix)):
        exists = True
        for j in range(len(matrix[i])):
            field = matrix[i][j]
            if field is not None:
                exists = False

        if exists:
            return '{}: {}'.format(days[i // 9], hours[i % 9])

    return -1



def hard_constraints_cost(matrix, data):
    """
    Calculates total cost of hard constraints: in every classroom is at most one class at a time, every class is in one
    of his possible classrooms, every teacher holds at most one class at a time and every group attends at most one
    class at a time.
    For everything that does not satisfy these constraints, one is added to the cost.
    :return: total cost, cost per class, cost of teachers, cost of classrooms, cost of groups
    """
    # cost_class: dictionary where key = index of a class, value = total cost of that class
    cost_class = {}
    for c in data.classes:
        cost_class[c] = 0

    cost_classrooms = 0
    cost_teacher = 0
    cost_group = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            field = matrix[i][j]                                        # for every field in matrix
            if field is not None:
                c1 = data.classes[field]                                # take class from that field

                # calculate loss for classroom
                if j not in c1.classrooms:
                    cost_classrooms += 1
                    cost_class[field] += 1

                for k in range(j + 1, len(matrix[i])):                  # go through the end of row
                    next_field = matrix[i][k]
                    if next_field is not None:
                        c2 = data.classes[next_field]                   # take class of that field

                        # calculate loss for teachers
                        if c1.teacher == c2.teacher:
                            cost_teacher += 1
                            cost_class[field] += 1

                        # calculate loss for groups
                        g1 = c1.groups
                        g2 = c2.groups
                        for g in g1:
                            if g in g2:
                                cost_group += 1
                                cost_class[field] += 1

    total_cost = cost_teacher + cost_classrooms + cost_group
    return total_cost, cost_class, cost_teacher, cost_classrooms, cost_group


def check_hard_constraints(matrix, data):
    """
    Checks if all hard constraints are satisfied, returns number of overlaps with classes, classrooms, teachers and
    groups.
    """
    overlaps = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            field = matrix[i][j]                                    # for every field in matrix
            if field is not None:
                c1 = data.classes[field]                            # take class from that field

                # calculate loss for classroom
                if j not in c1.classrooms:
                    overlaps += 1

                for k in range(len(matrix[i])):                     # go through the end of row
                    if k != j:
                        next_field = matrix[i][k]
                        if next_field is not None:
                            c2 = data.classes[next_field]           # take class of that field

                            # calculate loss for teachers
                            if c1.teacher == c2.teacher:
                                overlaps += 1

                            # calculate loss for groups
                            g1 = c1.groups
                            g2 = c2.groups
                            # print(g1, g2)
                            for g in g1:
                                if g in g2:
                                    overlaps += 1

    return overlaps
def get_empty_space_data(matrix, data):
    teachers_empty_space = {}
    groups_empty_space = {}
    subjects_order = {}

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            class_id = matrix[i][j]
            if class_id is None:
                continue
            cls = data.classes[class_id]

            # أوقات المدرسين
            if cls.teacher not in teachers_empty_space:
                teachers_empty_space[cls.teacher] = []
            teachers_empty_space[cls.teacher].append(i)

            # أوقات المجموعات
            for group in cls.groups:
                if group not in groups_empty_space:
                    groups_empty_space[group] = []
                groups_empty_space[group].append(i)

            # ترتيب الأنواع (P, V, L)
            key = (cls.subject, list(cls.groups)[0])
            if key not in subjects_order:
                subjects_order[key] = [-1, -1, -1]
            if cls.type == 'P':
                subjects_order[key][0] = i
            elif cls.type == 'V':
                subjects_order[key][1] = i
            elif cls.type == 'L':
                subjects_order[key][2] = i

    return teachers_empty_space, groups_empty_space, subjects_order
