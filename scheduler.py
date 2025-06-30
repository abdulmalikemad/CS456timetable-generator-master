import random
from operator import itemgetter
from utils import load_txt_data,generate_html_timetable, set_up, show_statistics, write_solution_to_file
from costs import check_hard_constraints, hard_constraints_cost, empty_space_groups_cost, empty_space_teachers_cost, \
    free_hour
from costs import get_empty_space_data

import copy
import math


def initial_population(data, matrix, free, filled, groups_empty_space, teachers_empty_space, subjects_order):
    classes = data.classes

    for index, classs in classes.items():
        random.shuffle(free)  # توزيع عشوائي

        for start_field in free:
            start_time = start_field[0]
            room = start_field[1]

            # ما فيش أي تحقق — نحجز أول ما نلقى مساحة
            if all((start_time + i, room) in free for i in range(int(classs.duration))):
                # حجز الزمن
                for i in range(int(classs.duration)):
                    free.remove((start_time + i, room))
                    filled.setdefault(index, []).append((start_time + i, room))
                    matrix[start_time + i][room] = index

                # تحديث الفجوات والمؤشرات
                for group_index in classs.groups:
                    for i in range(int(classs.duration)):
                        groups_empty_space[group_index].append(start_time + i)
                    insert_order(subjects_order, classs.subject, group_index, classs.type, start_time)

                for i in range(int(classs.duration)):
                    teachers_empty_space[classs.teacher].append(start_time + i)

                break  # انتقل للمادة اللي بعدها



def insert_order(subjects_order, subject, group, type, start_time):
    """
    Inserts start time of the class for given subject, group and type of class.
    """
    times = subjects_order[(subject, group)]
    if type == 'P':
        times[0] = start_time
    elif type == 'V':
        times[1] = start_time
    else:
        times[2] = start_time
    subjects_order[(subject, group)] = times


def exchange_two(matrix, filled, ind1, ind2):
    """
    Changes places of two classes with the same duration in timetable matrix.
    """
    fields1 = filled[ind1]
    filled.pop(ind1, None)
    fields2 = filled[ind2]
    filled.pop(ind2, None)

    for i in range(len(fields1)):
        t = matrix[fields1[i][0]][fields1[i][1]]
        matrix[fields1[i][0]][fields1[i][1]] = matrix[fields2[i][0]][fields2[i][1]]
        matrix[fields2[i][0]][fields2[i][1]] = t

    filled[ind1] = fields2
    filled[ind2] = fields1

    return matrix


def valid_teacher_group_row(matrix, data, index_class, row):
    """
    Returns if the class can be in that row because of possible teacher or groups overlaps.
    """
    c1 = data.classes[index_class]
    for j in range(len(matrix[row])):
        if matrix[row][j] is not None:
            c2 = data.classes[matrix[row][j]]
            # check teacher
            if c1.teacher == c2.teacher:
                return False
            # check groups
            for g in c2.groups:
                if g in c1.groups:
                    return False
    return True


def mutate_ideal_spot(matrix, data, ind_class, free, filled, groups_empty_space, teachers_empty_space, subjects_order):
    """
    Function that tries to find new fields in matrix for class index where the cost of the class is 0 (taken into
    account only hard constraints). If optimal spot is found, the fields in matrix are replaced.
    """

    # find rows and fields in which the class is currently in
    rows = []
    fields = filled[ind_class]
    for f in fields:
        rows.append(f[0])

    classs = data.classes[ind_class]
    ind = 0
    while True:
        # ideal spot is not found, return from function
        if ind >= len(free):
            return
        start_field = free[ind]

        # check if class won't start one day and end on the next
        start_time = start_field[0]
        end_time = start_time + int(classs.duration) - 1
        if start_time % 9 > end_time % 9:
            ind += 1
            continue

        # check if new classroom is suitable
        if start_field[1] not in classs.classrooms:
            ind += 1
            continue

        # check if whole block can be taken for new class and possible overlaps with teachers and groups
        found = True
        for i in range(int(classs.duration)):
            field = (i + start_time, start_field[1])
            if field not in free or not valid_teacher_group_row(matrix, data, ind_class, field[0]):
                found = False
                ind += 1
                break

        if found:
            # remove current class from filled dict and add it to free dict
            filled.pop(ind_class, None)
            for f in fields:
                free.append((f[0], f[1]))
                matrix[f[0]][f[1]] = None
                # remove empty space of the group from old place of the class
                for group_index in classs.groups:
                    groups_empty_space[group_index].remove(f[0])
                # remove teacher's empty space from old place of the class
                teachers_empty_space[classs.teacher].remove(f[0])

            # update order of the subjects and add empty space for each group
            for group_index in classs.groups:
                insert_order(subjects_order, classs.subject, group_index, classs.type, start_time)
                for i in range(int(classs.duration)):
                    groups_empty_space[group_index].append(i + start_time)

            # add new term of the class to filled, remove those fields from free dict and insert new block in matrix
            for i in range(int(classs.duration)):
                filled.setdefault(ind_class, []).append((i + start_time, start_field[1]))
                free.remove((i + start_time, start_field[1]))
                matrix[i + start_time][start_field[1]] = ind_class
                # add new empty space for teacher
                teachers_empty_space[classs.teacher].append(i+start_time)
            break


def evolutionary_algorithm(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order):
    """
    Evolutionary algorithm that tires to find schedule such that hard constraints are satisfied.
    It uses (1+1) evolutionary strategy with Stifel's notation.
    """
    n = 3
    sigma = 2
    run_times = 5
    max_stagnation = 200

    for run in range(run_times):
        print('Run {} | sigma = {}'.format(run + 1, sigma))

        t = 0
        stagnation = 0
        cost_stats = 0
        while stagnation < max_stagnation:

            # check if optimal solution is found
            loss_before, cost_classes, cost_teachers, cost_classrooms, cost_groups = hard_constraints_cost(matrix, data)
            if loss_before == 0 and check_hard_constraints(matrix, data) == 0:
                loss_after = loss_before
                print('Found optimal solution: \n')
                break


            # sort classes by their loss, [(loss, class index)]
            costs_list = sorted(cost_classes.items(), key=itemgetter(1), reverse=True)

            # 10*n
            for i in range(len(costs_list) // 4):
                # mutate one to its ideal spot
                if random.uniform(0, 1) < sigma and costs_list[i][1] != 0:
                    mutate_ideal_spot(matrix, data, costs_list[i][0], free, filled, groups_empty_space,
                                      teachers_empty_space, subjects_order)
                # else:
                #     # exchange two who have the same duration
                #     r = random.randrange(len(costs_list))
                #     c1 = data.classes[costs_list[i][0]]
                #     c2 = data.classes[costs_list[r][0]]
                #     if r != i and costs_list[r][1] != 0 and costs_list[i][1] != 0 and c1.duration == c2.duration:
                #         exchange_two(matrix, filled, costs_list[i][0], costs_list[r][0])

            loss_after, _, _, _, _ = hard_constraints_cost(matrix, data)
            if loss_after < loss_before:
                stagnation = 0
                cost_stats += 1
            else:
                stagnation += 1

            t += 1
            # Stifel for (1+1)-ES
            if t >= 10*n and t % n == 0:
                s = cost_stats
                if s < 2*n:
                    sigma *= 0.85
                else:
                    sigma /= 0.85
                cost_stats = 0

        print('Number of iterations: {} \nCost: {} \nTeachers cost: {} | Groups cost: {} | Classrooms cost:'
              ' {}'.format(t, loss_after, cost_teachers, cost_groups, cost_classrooms))


def simulated_hardening(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order, file):
    """
    Algorithm that uses simulated hardening with geometric decrease of temperature to optimize timetable by satisfying
    soft constraints as much as possible (empty space for groups and existence of an hour in which there is no classes).
    """
    # number of iterations
    iter_count = 2500
    # temperature
    t = 0.5
    _, _, curr_cost_group = empty_space_groups_cost(groups_empty_space)
    _, _, curr_cost_teachers = empty_space_teachers_cost(teachers_empty_space)
    curr_cost = curr_cost_group  # + curr_cost_teachers
    if free_hour(matrix) == -1:
        curr_cost += 1

    for i in range(iter_count):
        rt = random.uniform(0, 1)
        t *= 0.99                   # geometric decrease of temperature

        # save current results
        old_matrix = copy.deepcopy(matrix)
        old_free = copy.deepcopy(free)
        old_filled = copy.deepcopy(filled)
        old_groups_empty_space = copy.deepcopy(groups_empty_space)
        old_teachers_empty_space = copy.deepcopy(teachers_empty_space)
        old_subjects_order = copy.deepcopy(subjects_order)

        # try to mutate 1/4 of all classes
        for j in range(len(data.classes) // 4):
            index_class = random.randrange(len(data.classes))
            mutate_ideal_spot(matrix, data, index_class, free, filled, groups_empty_space, teachers_empty_space,
                              subjects_order)
        _, _, new_cost_groups = empty_space_groups_cost(groups_empty_space)
        _, _, new_cost_teachers = empty_space_teachers_cost(teachers_empty_space)
        new_cost = new_cost_groups  # + new_cost_teachers
        if free_hour(matrix) == -1:
            new_cost += 1

        if new_cost < curr_cost or rt <= math.exp((curr_cost - new_cost) / t):
            # take new cost and continue with new data
            curr_cost = new_cost
        else:
            # return to previously saved data
            matrix = copy.deepcopy(old_matrix)
            free = copy.deepcopy(old_free)
            filled = copy.deepcopy(old_filled)
            groups_empty_space = copy.deepcopy(old_groups_empty_space)
            teachers_empty_space = copy.deepcopy(old_teachers_empty_space)
            subjects_order = copy.deepcopy(old_subjects_order)
        if i % 100 == 0:
            print('Iteration: {:4d} | Average cost: {:0.8f}'.format(i, curr_cost))

    print('TIMETABLE AFTER HARDENING')
    
    print('STATISTICS AFTER HARDENING')
    show_statistics(matrix, data, subjects_order, groups_empty_space, teachers_empty_space)
    write_solution_to_file(matrix, data, filled, file, groups_empty_space, teachers_empty_space, subjects_order)
def cost_function(matrix, data):
    from costs import empty_space_groups_cost, empty_space_teachers_cost, subjects_order_cost, free_hour, get_empty_space_data
    teachers_empty_space, groups_empty_space, subjects_order = get_empty_space_data(matrix, data)
    _, _, group_cost = empty_space_groups_cost(groups_empty_space)
    _, _, teacher_cost = empty_space_teachers_cost(teachers_empty_space)
    order_score = subjects_order_cost(subjects_order)
    order_penalty = (100 - order_score) / 10
    gap_penalty = 1 if free_hour(matrix) == -1 else 0
    return group_cost + teacher_cost + order_penalty + gap_penalty

#_____________________________________________
def is_duplicated(matrix, cid):
    return sum(row.count(cid) for row in matrix) > 0


def generate_neighbor(matrix, data, filled):
    import copy
    neighbor = copy.deepcopy(matrix)
    filled_new = copy.deepcopy(filled)

    cls_id = random.choice(list(filled_new.keys()))
    cls = data.classes[cls_id]
    old_slots = filled_new[cls_id]

    for r, c in old_slots:
        neighbor[r][c] = None
    del filled_new[cls_id]

    for _ in range(50):
        day = random.randint(0, 5)
        slot_in_day = random.randint(0, 9 - cls.duration)
        start = day * 9 + slot_in_day
        room = random.choice(cls.classrooms)

        valid = True
        for i in range(cls.duration):
            r = start + i
            if neighbor[r][room] is not None:
                valid = False
                break
            for col in range(len(neighbor[0])):
                other_id = neighbor[r][col]
                if other_id is None:
                    continue
                other = data.classes[other_id]
                if other.teacher == cls.teacher or any(g in other.groups for g in cls.groups):
                    valid = False
                    break
            if not valid:
                break

        if valid:
            new_slots = []
            for i in range(cls.duration):
                r = start + i
                neighbor[r][room] = cls_id
                new_slots.append((r, room))

            if is_duplicated(neighbor, cls_id):
                continue

            filled_new[cls_id] = new_slots
            return neighbor, filled_new

    for r, c in old_slots:
        neighbor[r][c] = cls_id
    filled_new[cls_id] = old_slots
    return matrix, filled

#our work function Abdulmalik and mustfa 
#------------------------------------------------------------------------------------------------------------------
def simulated_annealing_final(data, initial_matrix, initial_filled, steps=2000, temp_init=100.0, cooling=0.99):
    import copy
    curr_matrix = copy.deepcopy(initial_matrix)
    curr_filled = copy.deepcopy(initial_filled)
    curr_cost = cost_function(curr_matrix, data)

    best_matrix = copy.deepcopy(curr_matrix)
    best_filled = copy.deepcopy(curr_filled)
    best_cost = curr_cost
    temp = temp_init

    for step in range(steps):
        temp *= cooling
        if temp < 0.01:
            break

        neighbor_matrix, neighbor_filled = generate_neighbor(curr_matrix, data, curr_filled)
        new_cost = cost_function(neighbor_matrix, data)
        delta = new_cost - curr_cost

        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temp):
            curr_matrix = neighbor_matrix
            curr_filled = neighbor_filled
            curr_cost = new_cost

            if curr_cost < best_cost:
                best_matrix = copy.deepcopy(curr_matrix)
                best_filled = copy.deepcopy(curr_filled)
                best_cost = curr_cost

        if step % 100 == 0:
            print(f"[{step:4}] Temp: {temp:.4f} | Cost: {curr_cost:.2f} | Best: {best_cost:.2f}")

    return best_matrix, best_filled
#------------------------------------------------------------------------------------------------------

import time

def main():
    filled = {}
    subjects_order = {}
    groups_empty_space = {}
    teachers_empty_space = {}
    file = 'ulaz1.txt'

    data = load_txt_data('test_files/' + file, teachers_empty_space, groups_empty_space, subjects_order)
    matrix, free = set_up(len(data.classrooms))

    # المرحلة 1: إنشاء الجدول المبدئي
    t1 = time.time()
    initial_population(data, matrix, free, filled, groups_empty_space, teachers_empty_space, subjects_order)
    t2 = time.time()
    print(f" Initial population time: {t2 - t1:.4f} seconds")
    generate_html_timetable(matrix, data.classrooms, data, output_file='initial_schedule.html')

    total, _, _, _, _ = hard_constraints_cost(matrix, data)
    print('Initial cost of hard constraints: {}'.format(total))

    # المرحلة 2: Evolutionary Algorithm
    t3 = time.time()
    evolutionary_algorithm(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order)
    t4 = time.time()
    print(f" Evolutionary algorithm time: {t4 - t3:.4f} seconds")
    generate_html_timetable(matrix, data.classrooms, data, output_file='after_evolutionary.html')

    print('STATISTICS AFTER EVOLUTIONARY')
    show_statistics(matrix, data, subjects_order, groups_empty_space, teachers_empty_space)

    # المرحلة 3: Simulated Annealing
    t5 = time.time()
    matrix, filled = simulated_annealing_final(data, matrix, filled)
    t6 = time.time()
    print(f"⏱️ Simulated annealing time: {t6 - t5:.4f} seconds")

    generate_html_timetable(matrix, data.classrooms, data, output_file='after_annealing.html')

    print('STATISTICS AFTER ANNEALING')
    teachers_empty_space, groups_empty_space, subjects_order = get_empty_space_data(matrix, data)
    show_statistics(matrix, data, subjects_order, groups_empty_space, teachers_empty_space)
    write_solution_to_file(matrix, data, filled, file, groups_empty_space, teachers_empty_space, subjects_order)

    print(f"✅ Total runtime: {t6 - t1:.4f} seconds")

if __name__ == '__main__':
    main()


