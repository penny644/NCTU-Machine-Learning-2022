import math

if __name__ == '__main__':
    a = int(input("Parameter a for the initial beta prior: "))
    b = int(input("Parameter b for the initial beta prior: "))
    file = open("testcase.txt", "r")
    case_count = 1
    for line in file.readlines():
        count_one = 0
        count_zero = 0
        count = 0
        for i in range(len(line)):
            if(line[i]!="\n"):
                count_one += int(line[i])
                count += 1
        count_zero = count - count_one
        probability = count_one / count
        bionomial = math.factorial(count) / (math.factorial(count_one) * math.factorial(count_zero)) * pow(probability, count_one) * pow(1-probability, count_zero)
        print("case %d: %s" %(case_count, line[0:count]))
        print("Likelihood: %.17f" %(bionomial))
        print("Beta prior:     a = %d b = %d" %(a,b))
        a += count_one
        b += count_zero
        case_count += 1
        print("Beta posterior: a = %d b = %d" %(a,b))
        print("")