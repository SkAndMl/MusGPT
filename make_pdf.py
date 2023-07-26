from fpdf import FPDF

class PDF(FPDF):

    def header(self):

        self.image("quill.jpeg", 10, 8, 25)
        self.set_font("helvetica", "B", 20)
        self.cell(0, 10, "Poetika", border=False, align='C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)

    def write_poem(self, poet,  text):
        text = poet + " says, \n" + text
        text = text.split("\n")
        self.set_font("times", 'BI', size=16)
        self.cell(100, 30, txt=text[0])
        self.ln()
        for line in text[1:]:
            self.set_font("times", "I", size=12)
            self.cell(100, 5, txt=" "*len(text[0])+line)
            self.ln()

def make_pdf(poet: str, text=""):

    pdf = PDF(orientation='P', unit="mm", format="letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.write_poem(poet, text)
    return pdf