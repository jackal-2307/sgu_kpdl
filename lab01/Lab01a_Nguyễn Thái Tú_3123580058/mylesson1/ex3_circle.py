import math

def calculate_circle_area(radius):
    area = math.pi * radius ** 2
    return area

def calculate_circumference(radius):
    circumference = 2 * math.pi * radius
    return circumference

if __name__ == "__main__":
    try:
        r = float(input("Nhập bán kính r của hình tròn: "))
        if r < 0:
            print("Bán kính phải lớn hơn hoặc bằng 0!")
        else:
            area = calculate_circle_area(r)
            circumference = calculate_circumference(r)
            print(f"Diện tích hình tròn với bán kính {r}: {area:.2f}")
            print(f"Chu vi hình tròn với bán kính {r}: {circumference:.2f}")
    except ValueError:
        print("Vui lòng nhập số hợp lệ cho bán kính!")