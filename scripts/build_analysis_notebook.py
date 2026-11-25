"""Build a concise analysis-only companion; the canonical notebook remains self-contained."""
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def build_analysis_notebook(root):
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s):cells.append(nbformat.v4.new_code_cell(s))
    md("""# تحلیل نتایج phi-CCT

این نوت‌بوک مخصوص کار تحلیلی و مرور نتایج است و مدل را دوباره اجرا نمی‌کند. از کد مشترک در `scripts/` و خروجی‌های کامل پروژه استفاده می‌کند. نوت‌بوک مرجع و مستقل برای تحویل، `phi_cct_esnli.ipynb` است.

۱۰۰۰ نمونهٔ ثابت، ۹۰۰۰ پاسخ واقعی و ۶۰۰۰ جفت معتبر داریم. insertion و deletion جدا تحلیل می‌شوند. نام‌موجودیت فرضیهٔ اصلی بوده؛ چهار بررسی دیگر ثانویه و از پیش مشخص‌شده‌اند.
""")
    code('''from pathlib import Path
import sys,json
import pandas as pd
from IPython.display import display,Image,Markdown
ROOT=Path.cwd().parent if Path.cwd().name=="notebooks" else Path.cwd()
if not (ROOT/"config/experiment.json").exists():raise RuntimeError("Open this notebook from the supplied project folder")
sys.path.insert(0,str(ROOT/"scripts"))
CONFIG=json.loads((ROOT/"config/experiment.json").read_text())
RECOMPUTE=True  # CPU only; retains the frozen primary analysis choices
''')
    md("## ۱. بازتولید محاسبات و کنترل خروجی خام\nتمام تطبیق labelها معنایی است. خطای تولید یا parsing نباید به صفر در E_D یا I_D تبدیل شود. بازنمونه‌گیری با واحد source_id انجام می‌شود و همهٔ editها و layoutهای یک نمونه را کنار هم نگه می‌دارد.")
    code('''from analyze_results import analyze,VARIABLES
from review_actual_examples import run_mention_sensitivity
from verify_real_outputs import verify_real_outputs
if RECOMPUTE:
    analyze(ROOT,CONFIG,"full")
    run_mention_sensitivity(ROOT,CONFIG)
verification=verify_real_outputs(ROOT)
display(pd.DataFrame([verification]).drop(columns=["checks"]))
''')
    md("## ۲. پوشش، دقت و معیار اصلی\nدقت مربوط به label مرجع ورودی اصلی است؛ برای editها، gold جدید مستقلی تعیین نشده است. تعداد پاسخ‌ها با تعداد نمونه‌های مستقل فرق دارد.")
    code('''coverage=pd.read_csv(ROOT/"outputs/results/coverage.csv")
display(coverage.completion_status.value_counts().to_frame("sources"))
display(pd.read_csv(ROOT/"outputs/results/accuracy.csv"))
summary=pd.read_csv(ROOT/"outputs/results/summary.csv")
display(summary[["family","n_sources","n_pairs","phi","phi_ci_low","phi_ci_high","impact_rate","mention_rate","n00","n01","n10","n11"]])
''')
    md("## ۳. هر پنج بررسی اجباری\nفرض صفر هر آزمون برابری phi در گروه‌های آن متغیر است. آزمون omnibus از bootstrap خوشه‌ایِ مرکز‌شده استفاده می‌کند؛ این آزمون تقریبی است و اثبات علیت نیست. BH روی هر ۱۰ آزمون برنامه‌ریزی‌شده اعمال شده است. زیرگروه با کمتر از ۲۰ source مستقل طبق قاعدهٔ ازپیش‌مشخص‌شده قابل آزمون نیست.")
    code('''tests=pd.read_csv(ROOT/"outputs/results/tests.csv")
display(tests[["family","variable","p_raw","p_bh","effect_max_abs_phi_difference","status","nonestimable_reason"]])
display(pd.read_csv(ROOT/"outputs/results/phi_contrasts.csv"))
for variable in VARIABLES:display(Image(filename=str(ROOT/f"outputs/figures/{variable}.png")))
display(Image(filename=str(ROOT/"outputs/figures/position_by_role.png")))
''')
    md("## ۴. تفسیر اصلی\nدر insertion، تنها تفاوت مربوط به named entity پس از BH معنادار باقی می‌ماند: phi گروه غیرموجودیت حدود 0.265 و گروه موجودیت حدود -0.158 است. گروه موجودیت در این طراحی فقط از French و Canadian تشکیل شده؛ نتیجه به همهٔ نام‌ها یا انواع موجودیت تعمیم داده نمی‌شود. برای IDF و موقعیت، شواهد خام با نتیجهٔ پس از اصلاح چندآزمونی فرق دارد. نبود معناداری، برابری قطعی را ثابت نمی‌کند. آزمون polarity در deletion به علت ۱۳ source منفی، مطابق طرح قابل‌برآورد نیست.")
    md("## ۵. تحلیل تکمیلیِ پس از مشاهدهٔ داده: اشارهٔ معنایی به ملیت\nاین بخش **post-hoc** است و جایگزین نتایج اصلی نمی‌شود. در Q8 مدل می‌گوید nationality، بدون تکرار Canadian. یک proxy محدود اضافه می‌کند: برای targetهای French/Canadian، اشارهٔ صریح به nationality/nationalities هم شمرده شود. ۲۰ نمونهٔ این proxy با متن و توضیح بررسی شده‌اند؛ این ارزیابی یک مطالعهٔ مستقل انسانی یا oracle عمومی معنایی نیست.")
    code('''display(pd.read_csv(ROOT/"outputs/results/mention_sensitivity.csv")[["level","mention_rule","n_sources","phi","phi_ci_low","phi_ci_high"]])
display(pd.read_csv(ROOT/"outputs/results/verbosity.csv"))
''')
    md("این proxy مقدار phi موجودیت را از حدود -0.158 به +0.257 می‌رساند؛ نزدیک به +0.265 گروه دیگر. این تغییر، حساسیت نتیجه به تعریف عملیاتیِ «اشاره» را نشان می‌دهد و سازوکار محتملی برای تفاوت اولیه فراهم می‌کند. اثر علّیِ کلی named entity یا بی‌اعتباری عمومی phi-CCT از این آزمایش اثبات نمی‌شود.")
    md("## ۶. مثال‌های کیفی قابل‌ردیابی\nتفسیر Q2 شامل یک مشاهدهٔ انسانی است. سایر تفسیرها حاصل بازبینی خروجی واقعی توسط Codex هستند و یک مطالعهٔ مستقل انسانی محسوب نمی‌شوند. ردیف failure قبل از inference حذف شده و E/I برای آن تعریف نمی‌شود.")
    code('''examples=pd.read_csv(ROOT/"outputs/qualitative/qualitative_examples.csv",keep_default_na=False)
for r in examples.to_dict("records"):
    display(Markdown(f"### {r['example_id']} — {r['case_type']}"))
    display(pd.Series({k:r.get(k,"") for k in ["source_id","edit_id","layout_id","target","original_premise","original_hypothesis","edited_premise","edited_hypothesis","baseline_explanation","edited_explanation","baseline_semantic_label","edited_semantic_label","E_D","I_D","interpretation","alternative_explanation","baseline_job_id","edited_job_id"]}).to_frame("value"))
''')
    md("## محدودیت‌های روش‌شناختی\nویژگی‌ها و روش editها هم‌زمان تغییر می‌کنند؛ گروه ملیت فقط دو واژه دارد؛ matching واژه‌ای paraphraseها را از دست می‌دهد؛ audit دستی پیش از اجرا کلی و محدود بوده و gold جدید تعیین نشده است. محدودیت ۲۵ کلمه یک درخواست در prompt است و رعایت‌نکردنش فیلتر نشده است. گزارش اصلی دو صفحه و جزئیات کامل در CSVها و نوت‌بوک مرجع هستند. هیچ مدل یا دیتاست اختیاری به عنوان اجرای انجام‌شده گزارش نمی‌شود.")
    nb=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    nbformat.validate(nb);nbformat.write(nb,root/"notebooks/analysis.ipynb")
    print("Created analysis-only companion notebook")


if __name__=="__main__":build_analysis_notebook(ROOT)
