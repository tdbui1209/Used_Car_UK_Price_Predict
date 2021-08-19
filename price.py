import streamlit as st
import pandas as pd
import pickle

st.title('Môi giới xe Four Man')

brand = st.selectbox('Chọn hãng xe:',
                    ('audi', 'mercedes', 'hyundai'))

if brand == 'audi':
    model = st.selectbox('Chọn model của xe:',
                        ('a1', 'a6', 'a4', 'a3', 'q3', 'q5', 'a5', 's4',
                         'q2', 'a7', 'tt', 'q7', 'rs6', 'rs3', 'a8', 'q8',
                         'rs4', 'rs5', 'r8', 'sq5', 's8','sq7', 's3',
                         's5', 'a2', 'rs7'))
elif brand == 'mercedes':
    model = st.selectbox('Chọn model của xe:',
                        ('slk', 's class', 'sl class', 'g class', 'gle class',
                         'gla class', 'a class', 'b class', 'glc class',
                         'c class', 'e class', 'gl class', 'cls class',
                         'clc class', 'cla class', 'v class', 'm class',
                         'cl class', 'gls class', 'glb class', 'x-class',
                         'clk', 'r class'))
else:
    model = st.selectbox('Chọn model của xe:',
                        ('i20', 'tucson', 'i10', 'ix35', 'i30', 'i40',
                         'ioniq', 'kona', 'veloster', 'i800', 'ix20',
                         'santa fe', 'accent', 'terracan', 'getz', 'amica'))

year = st.number_input('Nhập năm đăng ký xe:')
mileage = st.number_input('Nhập số miles đã đi:')
tax = st.number_input('Nhập thuế hàng năm:')
mpg = st.number_input('Nhập số mile/gallon (combined):')
engineSize = st.number_input('Nhập kích cỡ động cơ:')

transmission = st.selectbox('Chọn kiểu hộp số:',
                          ('Automatic', 'Manual', 'Semi-Auto', 'Other'))

fuelType = st.selectbox('Chọn kloại nhiên liệu:',
                       ('Petrol', 'Hybrid', 'Diesel', 'Other'))

final_model = pickle.load(open('final_model.sav', 'rb'))
lb_encoder = pickle.load(open('lb_encoder.sav', 'rb'))
OH_encoder = pickle.load(open('oh_encoder.sav', 'rb'))

test = pd.DataFrame({'model': [model], 'year': [year],
                     'mileage': [mileage], 'tax': [tax],
                     'mpg': [mpg], 'engineSize': [engineSize],
                     'transmission': [transmission],
                     'fuelType': [fuelType],
                     'brand': [brand]})

do_predict = st.button('Tính toán')
if do_predict:
    test['model'] = lb_encoder.transform(test['model'])
    cat_cols = ['transmission', 'fuelType', 'brand']

    cat_test = pd.DataFrame(OH_encoder.transform(test[cat_cols]))
    cat_test.index = test.index

    num_test = test.drop(cat_cols, axis=1)
    OH_test = pd.concat([num_test, cat_test], axis=1)

    OH_test.drop([3, 4, 1], axis=1, inplace=True)

    predict = final_model.predict(OH_test)
    st.write('Dự đoán giá của chiếc xe là:', predict[0])
