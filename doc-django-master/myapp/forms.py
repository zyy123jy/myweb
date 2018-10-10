from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Please name your file as test_data.txt',
        help_text='First line: APOE4, gender, education(years), APOE2; other lines: MMSE, ADAS-COG, CDRSB, Age'
    )
