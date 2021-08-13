import streamlit as st
import pandas as pd
import pickle

st.title('Môi giới xe Four Man')

brand = st.selectbox('Chọn hãng xe:',
                    ('Audi', 'Mercedes', 'Hyundai'))
model = st.text_input('Nhập model của xe (ví dụ: A6):')
model = st.selectbox('How would you like to be contacted?',
                    ('Email', 'Home phone', 'Mobile phone'))
year = st.number_input('Nhập năm đăng ký xe:')
mileage = st.number_input('Nhập số miles đã đi:')
tax = st.number_input('Nhập thuế hàng năm:')
mpg = st.number_input('Nhập số mile/gallon:')
engineSize = st.number_input('Nhập kích cỡ động cơ:')
transmission = st.text_input('Nhập kiểu hộp số:')
fuelType = st.text_input('Nhập loại nhiên liệu:')


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
