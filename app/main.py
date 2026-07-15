"""
main.py — MotionScope (MoveInsight Landing & Dashboard UI)
"""
from __future__ import annotations
import os, sys, time, tempfile, base64, io
from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

st.set_page_config(page_title="MoveInsight", layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS_PATH = APP_DIR / "assets" / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    
# Inyectar CSS adicional para mejorar selectbox y componentes
st.markdown("""
<style>
/* Selectbox styling */
[data-testid="stSelectbox"] div {
    background-color: white !important;
}
.ms-nav-link{
    transition: all .25s ease;
}
#titulo-sobre-sistema{
    color:#FFFFFF !important;
}

.ms-nav-link:hover{
    color:#0C8E8C !important;
    border-bottom:2px solid #0C8E8C;
}

.ms-nav-link[data-target="sec-upload"]:hover{
    color:white !important;
    background:#087979 !important;
}
            .ms-nav-link{
    transition: all .25s ease;
}

.ms-nav-link:hover{
    color:#0C8E8C !important;
    border-bottom:2px solid #0C8E8C;
}

.ms-nav-link[data-target="sec-upload"]:hover{
    color:white !important;
    background:#087979 !important;
}

.navbar-container{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:18px 0;
    border-bottom:1px solid #E2E8F0;
    margin-bottom:28px;
}

.navbar-links{
    display:flex;
    align-items:center;
    gap:36px;
}

@media(max-width:768px){
    .navbar-container{
        flex-direction:column;
        gap:18px;
    }
    .navbar-links{
        flex-wrap:wrap;
        justify-content:center;
        gap:14px;
    }
    #how-cards{
        flex-direction:column !important;
    }
}

[data-testid="stSelectbox"] select,
[data-testid="stSelectbox"] > div > div > div {
    background-color: white !important;
    border: 2px solid #008B8B !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-weight: 600 !important;
    color: #0F4C81 !important;
    cursor: pointer !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 4px rgba(0,139,139,0.08) !important;
}

[data-testid="stSelectbox"] select:hover,
[data-testid="stSelectbox"] > div > div > div:hover {
    border-color: #006666 !important;
    box-shadow: 0 4px 12px rgba(0, 139, 139, 0.2) !important;
}

[data-testid="stSelectbox"] select:focus,
[data-testid="stSelectbox"] > div > div > div:focus {
    border-color: #006666 !important;
    box-shadow: 0 0 0 3px rgba(0, 139, 139, 0.1) !important;
    outline: none !important;
}

/* Button styling - mejorado */
.stButton > button {
    background-color: #008B8B !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0,139,139,0.15) !important;
}

.stButton > button:hover {
    background-color: #006666 !important;
    box-shadow: 0 4px 16px rgba(0, 139, 139, 0.3) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    background-color: #005555 !important;
    transform: translateY(0) !important;
}

.stButton > button:disabled {
    background-color: #B8D4D4 !important;
    opacity: 0.6 !important;
    cursor: not-allowed !important;
}
</style>
""", unsafe_allow_html=True)
    
# Inyectar CSS adicional para mejorar los radio buttons (hacerlos parecer tarjetas)
st.markdown("""
<style>
/* Advanced styling to turn stRadio into a rich card list (prototype style) */
div.row-widget.stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
[data-testid="stRadio"] div[role="radiogroup"] > label {
    background-color: #FFFFFF !important;
    border: 1px solid #D1E4E4 !important;
    padding: 14px 18px 14px 68px !important;
    border-radius: 10px !important;
    cursor: pointer;
    transition: all 0.25s ease;
    margin: 0 !important;
    position: relative;
    display: flex !important;
    align-items: center !important;

    width: 100% !important;
    box-sizing: border-box !important;
}
[data-testid="stRadio"] label:hover {
    background-color: #F7FBFB !important;
    border-color: #008B8B !important;
    box-shadow: 0 2px 8px rgba(0, 139, 139, 0.08) !important;
}

[data-testid="stRadio"] label p {
    font-size: 1.02rem !important;
    font-weight: 800 !important;
    color: #0F4C81 !important;
    margin: 0 !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stRadio"] label p:last-of-type {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    margin-top: 2px !important;
}





/* Base icon styling */
[data-testid="stRadio"] label::before {
    content: "";
    position: absolute;
    left: 10px;
    top: 50%;
    transform: translateY(-50%);
    width: 44px;
    height: 44px;
    background-color: #E8F4F4;
    background-size: 60%;
    background-repeat: no-repeat;
    background-position: center;
    border-radius: 8px;
    display: block;
    transition: all 0.25s ease;
}


/* Base subtitle styling */
[data-testid="stRadio"] label p::after {
    content: " Análisis del movimiento";
    display: block;
    white-space: pre;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    margin-top: 4px;
}
[data-testid="stRadio"] label[data-checked="true"] p::after {
    color: rgba(255,255,255,0.85) !important;
}

/* Custom Subtitles & Icons for each option based on index */
[data-testid="stRadio"] label:nth-child(1) p::after { content: "Análisis de marcha frontal y posterior"; }
[data-testid="stRadio"] label:nth-child(1)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAADUUlEQVR4nO2dS3LrMAwEZdfb82K5U+rdKRfDCZLKwpuULetDisD09DYVWyIaI+pHL4sxhsttAdE+/n+/+lt8faLG4sGNXni6CNI7u6fwVBHuiyhnit/j/6sgKUCv4jWABHIC9C5aE5dATgADFmBUtzbhFJASwIAFGN2lTTQFZAQwx7AAcCwAHAsAxwLAsQBwZAQYffcuRO8OyghgjiElwKguDdHulxPA7EdOgN7dGsLdLylAz6KFePFlBehRvAAU/xfETvqpYLgAD/xeAOgQYLZhAeBYADgWAA5GAE8A4QKY51gAOBYAjgWA828RR/WFjl5IJ8DW4jewJLL3Ao4WNSB3AR/I7WyPbg6QBFI72jvKAyCCxA6OPIaHuAR39eK/K2C8+bv6BPFOLv4eCVRFKCnAloLsje74+rwR06CcAFsKf+a4HbA0KCVA764/8zkqEpSY4W4Z7LWinXkWoJ387uzc6ZFPT4O05vbqvJ5PArUXn+UEEOv6V1QudKlDwNpAzy5C/JFv9vZICvCKTIMdibZFUoC/XZZxwCPhNskIoDTImUktgIsPF8CMxwLAkRbAr4PBBTDvsQBwLAAcrACV7+D1RFYAFxguwBaaU0BTABcWLsAeGjwF5ASgFxQtwNHiN7A0UgIYsABnnyNs0BSQEWANP1cgLkCv5eAbMAXKC3DV62KqlBfgCE4BEQHc/XABRq0M0kBzgbICkIo0krICjF4apkEEKykApThXUE6AEWsDLQe/S4FyAlxNE5eglACjTvsCfLGolACzfiq2CadAGQGUizCTMgLMXh6uiQpYQgDVwc9AegGuvt4fsBRIL4ABCzDrbl+AUiC1AGuQz90RAszutICkQEoB/KAHXIAsBCAF0gng7ocLsEa2BaKbQAqkEkBhQKuRSoCsp30hnAJpBKg+kFVJI8Bap2W46BOiKZBKgKoDGQkELS1AlSVdI9G2SAlAGvBspBUga/HjyXZVPXSlEKDy4CkwXQADFkDxhxir4QSAk04Adz9EgMqTvxA6E0iXAAYsgOMfIkDVuFQkVQIYsADV4j9EJoKXC1BxkJRJkwAGIIAv/ebDCQBnugDVJn9qXCaA4uQvBM4EpieAgQjw7Dd7HP9gqkWlMcYYY4wSP8mLqmu6xRWTAAAAAElFTkSuQmCC"); }

[data-testid="stRadio"] label:nth-child(2) p::after { content: "Análisis de movilidad del hombro"; }
[data-testid="stRadio"] label:nth-child(2)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAACOklEQVR4nO3dUW7UMBSG0QzqezbGnhB7YmNeQVFfkajU5Mau73/OKxIi9uebjiYpxwEAAACkeB2Bzp+/3//3Z+PPr6g1ibnYzzY9OYb2F3hl45NC+HE0VrH5lX/Pd9Sy7Cc3bDSbBu0mwNOn9Ww2DdoFQHAAs07n2WgKtAlg9qacTSJoEwDBAaw6jWeDKdAiAK4TQLjtA1g9hs/NbwPbB8A9AggngHACCCeAcAIIt30Aq7+fH5s/H7B9ANwjgHAtAlg1hsfm479NAFzXJoDZp3E0OP2tApi5KaPJ5rcLgK9rF8DTp3M0Ov0fWl3Mk9/Vj2Yb33YCPLFpo+nmf2h7Yf/ydnBYAE88qjUaToKWt4CnntM7N3/+LyIAL4eGB8DXvB2LTmXH++mO6/RaPY6rL3DGfXos+Dc/FcKr20eunQI4v8Evrir/GeDqBnT8CXuHdSqfAGkbuULlFCidADZ/jsp19jEwXFkATv9cVetdEoDNX6Ni3d0Cwt0OwOlf6+76mwDhBBBOAOEEEE4A4QQQTgDhBBDudgAe7Vrr7vqbAOFKAjAF1qhY97IJIIK5qta79BYggjkq1/mxx8J9S1jviQPW6uWMmdGNJi+2+BQQrk0A/tu48AC4RgDhBBBOAOEEEE4A4QQQTgDhBBBOAOEEEE4A4QQQTgDhBBBOAOEEEE4A4QQQTgDhBBBOAOEEEE4A4QQQTgDh2gQw+2XN4eVQOmjxivPMF0VHk5MPAAAAAAAAAAAAAAAAwLb+Ai46s4i/Se8+AAAAAElFTkSuQmCC"); }

[data-testid="stRadio"] label:nth-child(3) p::after { content: "Análisis de balanceo del brazo"; }
[data-testid="stRadio"] label:nth-child(3)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAACP0lEQVR4nO3dUUrDQBQF0Lb4n425J3FPbiwrUAqCXxVKM5P35p7zK5LovXmTkDpeLgAAAECK69knUNn2/vn96Gv718cSv7slfohZoa9YhtvZJ9A9/Fe+r4K2zT3SdmCA3aZB/ATYDr56u02D6AJsg8LqVILoAhBcgNFX6dZkCkQWYFY4W4MSRBaAPwoQLq4As8fyVnwZiCsAwQWofjWeIaYAwg8ugPAfe7ssTPDBE0D4wQWoFP5e/PXwUktApeC7WGYCCD+4AFXD34uP/7vyJ9gx+C7ht54AlcPvpGUBqoe/N7n679qc6Ijg99+gkj8VfE0P/6hjdAu+1RIwI/xXQuwa/l3pE58V/CP+NjA4/BTXhDt54RcpwOzHN8EXKcAZz+3CL/IUIPzgCWDkB08A4QcXQPjBBaj+ooaBBTgrfKVb/F0A4yhAuNsqY/js43dlAoRTgHAKEE4BwilAuNsqr1/PPn5XJkA4BQh3W2EMG/+FJsDsMIRfcAmYFYrwC98DjA5H+A1uAkeFJPzgpwDhh24SJfjAAgg9sABCn6vdPQDHUoBwChBOAcIpQDgFCKcA4RQgnAKEU4BwChBOAcIpQDgFCKcA4RQgXLkC2OkjvAB3SjBP+a1ifUQscAIwT/m9gi0HY7XYK1gJxrEEhGuzWbQpMIYJEE4BwilAOAUIpwDhFCBcm13CvBMYwwQI12KbOFd/8DZxwh/LEhCu7H8Pd+UvNAGeDVP480zf3fu/aSB4AAAAAAAAAAAAAAC4POMHCAXZu4k+3pwAAAAASUVORK5CYII="); }

[data-testid="stRadio"] label:nth-child(4) p::after { content: "Flexo-extensión del codo"; }
[data-testid="stRadio"] label:nth-child(4)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAACgElEQVR4nO3cUY7TMBCA4Rbtuy/GnRB34mI+wSLeUKXtJq3tzHi+7xlVAf+eJE3o7QYAAAAAAABXaT9/f/rXn+fjFpSFX+N+S7Lw/c+vcMe6gxATwG6/zj3TwpsC4/2Y8JkkcmkAZ3e0U8V4JkBxlwdgChQPgGuFCMDVffEAznIxuGEApkDxAM4yBYoHwIYBuCUsHgDrhQvAFCgeAGuFDMAtYfEAznJLuGEApkDxAM4yBYoHwIYBuCUsHgDzhQ/AFCgeAHOlCMAtYfEAznJLuGEApkDxAM4yBY65V1pUUyR5ACN3thg2PwUcCan59ZF8AYzeua14BOkCeDWaZ+G0whGkC+CdxXoWQisawb3S4h/5vF7st4jSTIAZO7QfjGJnaQKY9ZSwF9vxKQMYtSuPfk4rNAVSBPDVrn1l9zrvbxDAuyrt8PQBPC7W464fdQ7vD59TJZLwAXzn1YWqssDbB8B7SgfQTIHaAfxTPYIQvxZ+tVY4gvQToPo3edsHcNXtWS8SVvgArlqsVuSNoZQBrPw6t20eQZox9903gs/+7Ch9w9PCvdKu90Zx4gBGjv4RIfRNpkG6v8SIFztGniJ68hBSHvyzBXRtUCCAGbu4FT0tpDvg/41etFbwv52lOdBnXlm42aeKniSCFAe54vqgagihDy6CtnkEYQ8smrZpCOEOKLI28CIxyuvpAgj4gGhlCAIIGsKqCFI+Do6iT1ykVY+hTYBBsj6CNgEGiXiFf0TKg642DfrEuEyAC36TKBIBTJQhAgEUJ4DiBBD8NOA2kKlMgAVe3cUrLiIFsMjZxfQsYEP9YASeBhbQgrwPAAAAAAAAAAAAAAAAAAAAAEBwfwHWkjDse/JErQAAAABJRU5ErkJggg=="); }

[data-testid="stRadio"] label:nth-child(5) p::after { content: "Inclinación y rotación del tronco"; }
[data-testid="stRadio"] label:nth-child(5)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAAB8klEQVR4nO3dwW0TQRSA4SXiThqiA1IQlyByoSCnAxqyKwBFyh1x2JkX/98n5ZqVJ7+erZld5zgAAACAik9nX+DLt59/zr7Gvbu9/jjt7/Rw1i/mYxBAnADiBBD38JE/wBTcTl4/EyAuH8D18vz97eeIygdQJ4A4AcQJIE4AcQKIWxKAzaC562YCxAkgTgBxAogTQJwA4gQQ9/kY6np5/npv13t8evl9DGMCxC29XWviMwLX95tBHp9efh2DrNo9NQHiBBAngDgBxAkgTgBxAogTQNzSANwaNm+dTIA4AcQJIM5x8DvHwSSNnQCrbp64Dj0OXsVngDgBxC0PwGbQrPUxAeIEECeAOAHECSBOAHECiBNAnADitn2V+7+eE1z9dPCEA64du6QmQJzj4IvjYMK8BcQJIE4AcQKIE0DctgDcGjZjPUyAOAHECSBOAHECiBt7GOTLotcwAeLGTgBPBwcmgM2g/evgLSBOAHECiBNAnADiBBAngDgBxAkgbnsA9d3A2+bXvz0A9hJAnADiBBAngDgBxAkgTgBxIwLYvRlSft0jAmAfAcQJIE4AcQKIE0CcAOIEEDcmgAmbIsXXOyYA9hBAnADiBBAngDgBxAkgTgBxAogbsRv1P/9P8B7chuwCvjEB4gQQJ4A4AQAAAEDHXxupaLmu2cjcAAAAAElFTkSuQmCC"); }

[data-testid="stRadio"] label:nth-child(6) p::after { content: "Análisis de la articulación de cadera"; }
[data-testid="stRadio"] label:nth-child(6)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAADOklEQVR4nO2dUW7bMBBELSL/vFjvVPROvRhP0MIfBoKgjUWKWs5w3/uMDWl3drSkKNF5PAAAAAAAAAAAAAAAAAAAYEOOiJPUH7/+vPtO+/0zJBYlqoAux8rkMpqhiulyqCS3sxmqsC7lIZzknceNoorrcjgVyKkbVBNditPV6dINqpEuxa0Y6iaoZroUxyKsPr9qXCPnL25JqsWhFk9vHMUxSbV4qkgcI/FMuw0ET4qry1Xiqua60AGSU5xd7nrrpRQfHSA5GCA5xb3NOS6/KsVJB0gOBkgOBkjOx2MjXMZlJegAycEAycEAycEAySk7vYgJ/fWjAyQHAyTnlAEYBjw5Uzc6QHJOG4Au4MXZetEBktNlALqABz116u4AmECb3voMDQGYQJORugzPATCBFqP1uDQJxAQaXKnDtHV+XsaIZ8YFOO02kG4Qyyy95X8lbIfO0ibls+RXwnoD/leQV34PbzcDzNBlRk1CXwpleNDVhaXg5GCA5GCA5CzdGPLdZEZhfMyQ76mTzpx1jhwvE22ydu+OF9oBKPx5jaI6QtgcgOJr6nXLENCz+jVjocOJ1pnv1VXE5UPA/4L+LrDXZzsZoQ3m+/zbncNB+Gvhz2OdPd4udwKtI9/ZWr/7zm0dYNbV+0zCuRO0CQW9M//iIIZrJ2gG+bIvYFPk9gVcdbVbF2gm+X44j8dKpqimuvAwKDlhG0OUrhAH6qBeIRtDRsEEejqF7wwaTc7NPHVBnqE7g64wo5hKE0CHBR+5rWE9ybpd/StytNwaFjXRiaItmiAv+9exV0/+LvnnZ65X/9kcVhb/yaH24Ofd8VSv/s+ciX+2Zik2hzoU300Pm82hTsV30uNQd75b4d00OVzan6MRqoEOktvDnc1QzfK+VUiXiVDmPEPEi7qXjzRD3SSn7QRTeXjlYujQ9um+qhdJVDdbMn5iBJ05zZLHwQoTN0XaAl2WF4Ju8Fh6QSw3QGYjNIFOuDyAjEZoAoV/IRNIBiM0ocK/kAtoRyM0wcK/kA3M3QxNuOifsQjSxQzNpOifsQtYyRDNsOBfsU8g4x4EAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABIzV+iLPo/OM89fgAAAABJRU5ErkJggg=="); }

[data-testid="stRadio"] label:nth-child(7) p::after { content: "Análisis de flexo-extensión de rodilla"; }
[data-testid="stRadio"] label:nth-child(7)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAACaUlEQVR4nO3c3ZHaUBAFYUwCNzHn5HJOm9hEYBevlM3q90pzur/nZSWYvhIUQo+HJEmScH48IMbP33+2PK6+fkW/Rs+rd0DXMgA4A4AzADgDgDMAOAOAMwA4A4AzADgDgDMAOAOAMwA4A4AzADgDgDMAOAOAMwA4A4AzADgDgDMAOAOAMwA4A4AzADgDgDMAOAOAMwA4A4AzADgDgEMEsPX+QATRN0A6cvAVerOoyCd15oqvsBDiTgFnH+5H2OkkKoBZwxlBEcQEMHsoIySCiAA+DeN1zt5z3q4Pj02I4Jk+/CO2UcERtA7gfy/+3lW/9n92jqB1AEt1HtDZnomrf8nfbd1OhR0F2gYwY/hrI+ioZQBLBnv0ihwXbHOGlgH8y+xVWSFHgZgAZqzE0XCFxwVw9yGMm+9f+wC+OxzP/DKoAk4DEQFoOwOAMwA4A4AzADgDgIsIYOZHs5r4kXOGdgHc/bN33Xz/2gdw5RCq2XBRAXhRKCiAJSvxjEvCHpO3OUPLAGZfuFEnXWhyB20DWHpp1t4IauHwO67+1gGssfUq4Wo6VEwAn44CZ1wSNsJWf/sAZv1oY0z48clVWu/8mmG/BnX0UaGaD/+l/RO46t15BQw/4hRwxVAqZPhxAcz+MihB1JN55z2C4AEcEUKFrfh30U9ubwgVPvzI9wBaxwDgDADOAOAMAM4A4AwAzgDgDADOAOAMAM4A4AwAzgDgDADOAOAMAM4A4AwAzgDgDADOAOAMAM4A4AwAzgDgDADOAOAMAM4A4AwAzgDgDADOAOAMAM4A4AwAzgDgDADOAOAMAM4AJEmSJJq/+0XU5S0TNTcAAAAASUVORK5CYII="); }

[data-testid="stRadio"] label:nth-child(8) p::after { content: "Análisis de miembro inferior completo"; }
[data-testid="stRadio"] label:nth-child(8)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAAC7ElEQVR4nO2dW04DMRAEFy7gi3EnxJ24mE8AykckhACRfbh7pqv+o8yMy7N+EHbbAAAAACCOJ3UA4+XtYwtnvr/KxuF5E8Lg6+sgE4DB96iHRAAG36cuywVg8L3qI10DgB4ECAcBwkGAcBAgHAQIBwHCQYBwECCc56SbrwrMxfWRdAAk8KmL7BGABB71kK4BkEBfh6dOt2MrCzlM43oUdgHhIEA41gJ0+eORYZyHtQBwPW0EWL3QmsYLu0gBYB8IEA4ChIMA4SBAOLYCOO+dO+VjKwCsoYUAqj35bHAW0EIA2A8ChIMA4SBAOAgQDgKEYymA66FJx7wsBYB1lBdAfRgzix8GlRcAjoEA4SBAOAgQDgKEgwDh2AngeFjSOT87AWAtpQVwOYSZJnHECQDHQYBwECAcBAgHAURbQZftoM3q9UhBHFbho2j8Fh3g6GxQz6ZROH65AHuT/z5rVEUcxeMv/eJIdRFH8fgtOsDR56DD879y/DYCOBQjEYtXxx4d+K+fX9VGR/H4LTtAdWbBDoYA4SBAOAgQjsUbQ+4Ln/8ugP7aP696Ds9fFm7//f6vn1HEf4cOEI6NAHu3P+p7gOrxl35n0Pfiqf9j+HhwMNXx37DYt54xC5R78FE4fotHwJknaQpm4fgtOsDemaQe+A5vE7PoAI9yK5xD8X7CNa5WAsB5IEA4CBCOjQB7j4EdmTuOg7d0AUADAoSDAOEgQDgIEA4ChIMA4SBAOAgQDgKEgwDhWAjQ6R6g2n2AhQCgAwHCQYBwECAcBAgHAcJBgHAQIBwECAcBwpEL0PEYuNJxcKmfVFeSYBTJb/kXVv4tfcf8lj4Czmp16hu0TvktE+DspNwkGEXzWyLAVcm4SDAK5yffBYCWywW42mJ1FxjF86MDhIMA4SBAOJcLcPWhhvpQaBbPjw4QzhIBrrJYPfs75LesA5ydjMvgV89v6SPgrKTcBr9yflwHXwTXwScVy3W2P0L3/AAAAACgGJ9c01jzvo4suwAAAABJRU5ErkJggg=="); }

[data-testid="stRadio"] label:nth-child(9) p::after { content: "Análisis de postura cervical"; }
[data-testid="stRadio"] label:nth-child(9)::before { background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAACVklEQVR4nO3c21EDMRAF0YUESIycKHIiMUUAGVA8Vp571X2+qWJn1JbNB74uSVxPF8jL6/vnT392fbwhdnP0kL85cGoQxw1156ETYjhikEcc+qkxPF/lJg8/4ff/V229iYtfhbdB5Q2QePjJz/WdqmKbFrxKboOaG6Dp8JuetyKAlmU2Pnd8AA1LbH7+6ADSl3fCHLEBJC/tpHkiA0hd1olzRQYgcACJr5KT54sKIG05hDmjAhA4gKRXBWneiABSlkGcOyIAzTEAuPEAEq5B8vzjAWiWAcA9k6+/FJN78AaAMwA4A4AbC8D3/4x9eAPAGQCcAcAZAJwBwBkAnAHAGQCcAcAZANxYAC1foHD6PrwB4AwAzgDgRgPwc8D8HrwB4AwAbjwA+tvAGp5/PADNMgC4iACmr0Hy3BEBpCyDOG9MAJoRFUDKq4I0Z1QAacshzBcXgB4rMoC0V8nJc0UGkLqsE+eJDSB5aSfNER1A+vJOeP74ABqW2PzcFQG0LLPxeSsesunLJVbJwdfdAA1LXqHP9Z26B068DVbhwVffALqPAcAZAJwBwBkAnAHAGQCcAcAZAJwBwBkAnAHAGQCcAcAZAJwBwBkAnAHAGQCcAcAZAJwBwBkAnAHAGQCcAcAZAJwBwD0R/nmz3dr4z6dbbwAPP3+P2wLw8Dv2uSUAD3+PHXu9PQAPf6+79+tfAXAGAGcAcAYAZwBwBgBnAHAGAGcAcAYAZwBwBgB3ewDN35zd4O79brkBjGCPHXvd9hZgBB373PoZwAju4R4lSZIkSZIkSdL1R1+EcbiScZjWtAAAAABJRU5ErkJggg=="); }

/* ============================
   RADIO CARDS
============================ */


[data-testid="stRadio"] label {
    position: relative;
    display: flex !important;
    align-items: center !important;

    width: 100% !important;
    box-sizing: border-box !important;

    background: white;
    border: 2px solid #D6E7E8;
    border-radius: 14px;

    padding: 18px 18px 18px 72px !important;
    margin-bottom: 12px !important;

    transition: all .25s ease;
    cursor: pointer;
}

/* Oculta el radio original de Streamlit */
[data-testid="stRadio"] input[type="radio"] {
    position: absolute;
    opacity: 0;
    pointer-events: none;
}

[data-baseweb="radio"],
[data-testid="stRadio"] svg,
[data-testid="stRadio"] [role="radio"] {
    display: none !important;
}

/* Hover */
[data-testid="stRadio"] label:hover {
    border-color: #0C8E8C;
    box-shadow: 0 5px 14px rgba(12,142,140,.12);
}

/* Tarjeta seleccionada */
/* Tarjeta seleccionada */
[data-testid="stRadio"] label:has(input:checked) {
    background-color: #0C8E8C !important;
    border: 2px solid #0C8E8C !important;
    box-shadow: 0 6px 18px rgba(12,142,140,.25);
}

/* Todo el texto en blanco */
[data-testid="stRadio"] label:has(input:checked) * {
    color: white !important;
}

/* Subtítulo */
[data-testid="stRadio"] label:has(input:checked) p::after {
    color: rgba(255,255,255,.85) !important;
}

/* Icono */
[data-testid="stRadio"] label:has(input:checked)::before {
    background-color: rgba(255,255,255,.18);
    filter: brightness(0) invert(1);
}

/* Check */
[data-testid="stRadio"] label:has(input:checked)::after {
    content: "✓";
    position: absolute;
    right: 18px;
    top: 50%;
    transform: translateY(-50%);
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: white;
    color: #0C8E8C;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
}
/* ==========================================================
   RESPONSIVE (TABLETS Y CELULARES)
========================================================== */

@media screen and (max-width: 768px){

    /* Espaciado general */
    .block-container{
        padding-left:1rem !important;
        padding-right:1rem !important;
        padding-top:1rem !important;
    }

    /* Todas las columnas de Streamlit pasan a vertical */
    [data-testid="stHorizontalBlock"]{
        flex-direction:column !important;
        gap:1rem !important;
    }

    [data-testid="column"]{
        width:100% !important;
        flex:100% !important;
    }

    /* ================= NAVBAR ================= */

    #sec-inicio > div{
        flex-direction:column !important;
        align-items:center !important;
        gap:18px !important;
        text-align:center;
    }

    /* Logo */
    #sec-inicio img{
        height:55px !important;
        width:auto !important;
    }

    /* Menú */
    #sec-inicio > div > div:nth-child(2){
        display:flex !important;
        flex-wrap:wrap !important;
        justify-content:center !important;
        gap:12px !important;
    }

    .ms-nav-link{
        font-size:.9rem !important;
        padding:6px 4px !important;
    }
    [data-testid="stWidgetLabel"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Botón */
    .ms-nav-link[data-target="sec-upload"]{
        width:100%;
        text-align:center;
    }

    /* ================= HERO ================= */

    h1{
        font-size:2.1rem !important;
        text-align:center !important;
        line-height:1.2 !important;
    }

    h3{
        text-align:center !important;
    }

    p{
        text-align:center !important;
    }

    img{
        max-width:100%;
        height:auto !important;
    }

}
</style>
""", unsafe_allow_html=True)

# ── Load Logo ─────────────────────────────────────────────────────────────────
LOGO_PATH = APP_DIR / "assets" / "final_logo.png"
if LOGO_PATH.exists():
    with open(LOGO_PATH, "rb") as f:
        b64_logo = base64.b64encode(f.read()).decode()
    LOGO_HTML = f'<img src="data:image/png;base64,{b64_logo}" height="100" style="margin-right:10px; vertical-align:middle;">'
else:
    LOGO_HTML = '<div style="font-size: 2rem; color: #0F4C81; margin-right: 12px;">🏃</div><div style="font-size: 1.4rem; font-weight: 800; color: #0F4C81;">MoveInsight</div>'

# ── Load Hero Image ───────────────────────────────────────────────────────────
HERO_PATH = APP_DIR / "assets" / "hero.png"
if HERO_PATH.exists():
    with open(HERO_PATH, "rb") as f:
        b64_hero = base64.b64encode(f.read()).decode()
    HERO_HTML = f'<img src="data:image/png;base64,{b64_hero}" style="width: 100%; max-height:420px; height:auto; object-fit:cover; border-radius:24px; box-shadow:0 10px 25px rgba(0,0,0,.05); object-fit: cover; border-radius: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.05);">'
else:
    HERO_HTML = '<div style="background: #E0F2FE; height: 500px; border-radius: 24px;"></div>'

# ── Safe HTML Wrapper ──
def safe_html(html_str: str) -> str:
    """Removes empty lines to prevent Streamlit from breaking HTML tags into markdown paragraphs."""
    lines = [line for line in html_str.split('\n') if line.strip() != '']
    return "<div class='safe-html'>" + " ".join(lines) + "</div>"

# ── Module registry ───────────────────────────────────────────────────────────
@st.cache_resource
def _load_modules():
    from analysis.gait     import GaitModule
    from analysis.knee     import KneeModule
    from analysis.hip      import HipModule
    from analysis.trunk    import TrunkModule
    from analysis.shoulder import ShoulderModule
    from analysis.neck     import NeckModule
    from analysis.arm      import ArmModule
    from analysis.elbow    import ElbowModule
    return {
        "Cuerpo Completo / Marcha": GaitModule(),
        "Hombro":          ShoulderModule(),
        "Brazo":           ArmModule(),
        "Codo":            ElbowModule(),
        "Tronco":          TrunkModule(),
        "Cadera":          HipModule(),
        "Rodilla":         KneeModule(),
        "Piernas / Marcha inferior": KneeModule(),
        "Cuello":          NeckModule(),
    }

MODULES = _load_modules()
MODULE_ICONS = {
    "Cuerpo Completo / Marcha": "", "Hombro": "", "Brazo": "", 
    "Codo": "", "Tronco": "", "Cadera": "", "Rodilla": "", "Cuello": ""
}

def _reset():
    for k in ("video_path","summary","video_info","analysis_name",
              "summary_text","pdf_bytes", "uploaded_file", "selected_choice"):
        st.session_state.pop(k, None)

# ==============================================================================
# LANDING PAGE VIEW
# ==============================================================================
if "video_path" not in st.session_state and st.session_state.get("uploaded_file") is None:
    st.markdown('<style>[data-testid="stSidebar"] { display: none !important; }</style>', unsafe_allow_html=True)
    
    nav_html = f"""
        <div id="sec-inicio" class="navbar-container">
        <div style="display:flex; align-items:center;">{LOGO_HTML}</div>
        <div class="navbar-links">
            <a href="#" class="ms-nav-link" data-target="sec-inicio" style="color:#0C8E8C; font-weight:700; font-size:0.95rem; text-decoration:none; border-bottom:2px solid #0C8E8C; padding-bottom:4px;">Inicio</a>
            <a href="#" class="ms-nav-link" data-target="sec-caracteristicas" style="color:#334155; font-weight:600; font-size:0.95rem; text-decoration:none;">Características</a>
            <a href="#" class="ms-nav-link" data-target="sec-tipos" style="color:#334155; font-weight:600; font-size:0.95rem; text-decoration:none;">Tipos de Análisis</a>
            <a href="#" class="ms-nav-link" data-target="sec-como-funciona" style="color:#334155; font-weight:600; font-size:0.95rem; text-decoration:none;">Cómo Funciona</a>
            <a href="#" class="ms-nav-link" data-target="sec-sobre-sistema" style="color:#334155; font-weight:600; font-size:0.95rem; text-decoration:none;">Sobre el Sistema</a>
        </div>
        <a href="#" class="ms-nav-link" data-target="sec-upload" style="background:#0C8E8C; color:white; padding:10px 22px; border-radius:8px; font-weight:700; font-size:0.9rem; text-decoration:none; white-space:nowrap;">Comenzar Análisis &rarr;</a>
    </div>
    """

    st.markdown(safe_html(nav_html), unsafe_allow_html=True)

    st.components.v1.html("""
    <script>
    function wireNavLinks() {
        const doc = window.parent.document;
        const links = doc.querySelectorAll('.ms-nav-link');
        links.forEach(function(link) {
            if (link.dataset.wired === "true") return;
            link.dataset.wired = "true";
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = link.getAttribute('data-target');
                const target = doc.getElementById(targetId);
                if (target) {
                    target.scrollIntoView({behavior: 'smooth', block: 'start'});
                }
            });
        });
    }
    wireNavLinks();
    setTimeout(wireNavLinks, 400);
    setTimeout(wireNavLinks, 1200);
    </script>
    """, height=0)

    col_text, col_img = st.columns([1, 1], gap="large")
    with col_text:
        st.markdown(safe_html('<div style="background: #E0F2FE; color: #0284C7; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; display: inline-block; margin-bottom: 24px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: middle; margin-right:6px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>Tecnología de Visión por Computador + Inteligencia Artificial</div>'), unsafe_allow_html=True)
        st.markdown(safe_html('<h1 style="font-size: 3.5rem; line-height: 1.1; color: #0F4C81; margin-bottom: 24px; letter-spacing: -0.03em;">Análisis de Movimiento<br><span style="color: #008B8B;">Inteligente y Profesional</span></h1>'), unsafe_allow_html=True)
        st.markdown(safe_html('<p style="font-size: 1.1rem; color: #64748B; line-height: 1.6; margin-bottom: 32px; max-width: 90%;">MoveInsight es un sistema avanzado que utiliza IA para analizar los movimientos del cuerpo humano a partir de videos, generando métricas objetivas que apoyan la evaluación del rendimiento físico y la prevención de lesiones.</p>'), unsafe_allow_html=True)
        
        st.markdown(safe_html('<div id="sec-upload" style="background: #F0F9F9; border: 2px solid #008B8B; padding: 28px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,139,139,0.08);"><div style="font-weight: 700; color: #008B8B; margin-bottom: 16px; font-size: 1.1rem;">Comienza tu análisis:</div></div>'), unsafe_allow_html=True)
        
        landing_upload = st.file_uploader("Arrastra tu video aquí", type=["mp4","avi","mov"], label_visibility="collapsed")
        
        # Botón para mostrar/ocultar lista de análisis
        if "show_analysis_list" not in st.session_state:
            st.session_state.show_analysis_list = False
        
        arrow = "  ▴" if st.session_state.show_analysis_list else "  ▾"
        toggle_btn = st.button(
            f"Seleccionar Análisis {arrow}",
            use_container_width=True,
            key="toggle_analysis_list"
        )
        if toggle_btn:
            st.session_state.show_analysis_list = not st.session_state.show_analysis_list
            st.rerun()
        
        # Radio buttons con estilos personalizados (mostrar solo si está abierto)
        options = list(MODULES.keys())
        if st.session_state.show_analysis_list:
            landing_choice = st.radio(
                "Análisis",
                options,
                label_visibility="collapsed",
                key="landing_analysis_radio"
            )
            st.session_state.selected_landing_choice = landing_choice
        else:
            landing_choice = st.session_state.get("selected_landing_choice", options[0])
        
        # Botón de análisis
        analyze_btn = st.button("🚀 Comenza Análisis", use_container_width=True, disabled=(landing_upload is None), key="landing_analyze")
        
        if analyze_btn:
            st.session_state["uploaded_file"] = landing_upload
            st.session_state["selected_choice"] = landing_choice
            st.rerun()
            
        st.markdown(safe_html('<div style="margin-top: 24px; font-size: 0.85rem; color: #94A3B8;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: middle; margin-right:4px;"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>Tus videos son privados y seguros. No almacenamos tu información.</div>'), unsafe_allow_html=True)

    with col_img:
        st.markdown(safe_html(HERO_HTML), unsafe_allow_html=True)

    st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)

    # ── Features ──

    st.markdown('<div id="sec-caracteristicas"></div>', unsafe_allow_html=True)

    st.markdown(
        safe_html(
            '<h3 style="text-align: center; color:#0F4C81; margin-bottom:32px; font-size:1.5rem;">Características de la Plataforma</h3>'
        ),
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4, gap="large")
    svg_icons = [
        '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#008B8B" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#008B8B" stroke-width="2"><rect x="18" y="3" width="4" height="18"/><rect x="10" y="8" width="4" height="13"/><rect x="2" y="13" width="4" height="8"/></svg>',
        '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#008B8B" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#008B8B" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>'
    ]
    features = [
        (svg_icons[0], "Análisis Preciso", "Detección de 17 puntos clave del cuerpo con IA de última generación para máxima precisión."),
        (svg_icons[1], "Métricas Objetivas", "Índices biomecánicos y rangos de referencia orientativos basados en principios científicos."),
        (svg_icons[2], "Reportes Profesionales", "Reportes claros con interpretación de resultados para apoyar la toma de decisiones."),
        (svg_icons[3], "Privacidad Garantizada", "Tus videos y datos son privados. No compartimos ni almacenamos información personal.")
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(safe_html(f"""
            <div style="background: white; padding: 24px; border-radius: 16px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px rgba(0,0,0,0.02); height: 100%;">
                <div style="margin-bottom: 16px;">{icon}</div>
                <div style="font-weight: 700; color: #0F4C81; font-size: 1.1rem; margin-bottom: 8px;">{title}</div>
                <div style="color: #64748B; font-size: 0.9rem; line-height: 1.5;">{desc}</div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)

    # ── How it works ──
    how_it_works = """
    <div id="sec-como-funciona" style="background: #F8FAFC; padding: 40px; border-radius: 24px; border: 1px solid #E2E8F0;">
        <h3 style="text-align: center; color: #0F4C81; margin-bottom: 40px; font-size: 1.5rem;">¿Cómo funciona?</h3>
        <div id="how-cards" style="display: flex; justify-content: space-between; gap: 24px;">
            <div style="flex: 1;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                    <div style="background: #008B8B; color: white; width: 24px; height: 24px; border-radius: 12px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size: 0.8rem;">1</div>
                    <div style="font-weight: 700; color: #0F4C81;">Sube tu video</div>
                </div>
                <div style="color: #64748B; font-size: 0.9rem; padding-left: 36px;">Carga un video en formato MP4 de la actividad que deseas analizar.</div>
            </div>
            <div style="flex: 1;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                    <div style="background: #008B8B; color: white; width: 24px; height: 24px; border-radius: 12px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size: 0.8rem;">2</div>
                    <div style="font-weight: 700; color: #0F4C81;">Selecciona el análisis</div>
                </div>
                <div style="color: #64748B; font-size: 0.9rem; padding-left: 36px;">Elige la parte del cuerpo o el tipo de análisis que deseas realizar.</div>
            </div>
            <div style="flex: 1;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                    <div style="background: #008B8B; color: white; width: 24px; height: 24px; border-radius: 12px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size: 0.8rem;">3</div>
                    <div style="font-weight: 700; color: #0F4C81;">Procesamiento con IA</div>
                </div>
                <div style="color: #64748B; font-size: 0.9rem; padding-left: 36px;">Nuestro sistema analiza el movimiento y calcula métricas clave en tiempo real.</div>
            </div>
            <div style="flex: 1;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                    <div style="background: #008B8B; color: white; width: 24px; height: 24px; border-radius: 12px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size: 0.8rem;">4</div>
                    <div style="font-weight: 700; color: #0F4C81;">Resultados y reporte</div>
                </div>
                <div style="color: #64748B; font-size: 0.9rem; padding-left: 36px;">Obtén tu video procesado y un reporte profesional con los resultados al instante.</div>
            </div>
        </div>
    </div>
    """
    st.markdown(safe_html(how_it_works), unsafe_allow_html=True)
    
    st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)
    
    # ── Cards explaining each analysis type ──

    st.markdown('<div id="sec-tipos"></div>', unsafe_allow_html=True)

    st.markdown(
        '<h3 style="text-align: center; color: #0F4C81; margin-bottom: 32px; font-size: 1.5rem;">Tipos de Análisis Disponibles</h3>',
        unsafe_allow_html=True
    )

    desc_map = {
        "Cuerpo Completo / Marcha": "Analiza la simetría y fluidez de la marcha. Evalúa el movimiento global del cuerpo en conjunto.",
        "Hombro": "Mide el rango de movilidad (ROM) del hombro, útil para rehabilitar el manguito rotador o capsulitis.",
        "Brazo": "Calcula el balanceo del brazo, un indicador clave para la eficiencia de la carrera y marcha.",
        "Codo": "Evalúa los grados de flexo-extensión, crítico para recuperación deportiva de miembros superiores.",
        "Tronco": "Mide la inclinación y rotación, detectando desviaciones posturales asimétricas durante el movimiento.",
        "Cadera": "Analiza la articulación de la cadera, fundamental para la estabilidad pélvica y marcha patológica.",
        "Rodilla": "Mide ángulos de flexión, identificando anomalías o posibles déficits estructurales.",
        "Piernas / Marcha inferior": "Análisis de miembro inferior completo para detectar alteraciones en la marcha.",
        "Cuello": "Cuantifica la postura cervical y rango de movimiento de la cabeza para evitar tensiones."
    }
    
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    for i, opt in enumerate(options):
        col = [c1, c2, c3, c4][i % 4]
        with col:
            st.markdown(safe_html(f"""
            <div style="background: white; border: 1px solid #E2E8F0; padding: 20px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); min-height: 140px;">
                <div style="color: #0F4C81; font-weight: 700; margin-bottom: 8px;">{opt}</div>
                <div style="color: #64748B; font-size: 0.85rem; line-height: 1.4;">{desc_map[opt]}</div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    # ── Sobre el Sistema ──
    about_html = """
    <div  style="background: linear-gradient(135deg, #0F4C81 0%, #0C8E8C 100%); padding: 48px; border-radius: 24px; color: white;">
        <h4 id="titulo-sobre-sistema" style="margin-bottom: 20px; font-size: 1.6rem; color: white !important;">Sobre el Sistema</h3>        
        <p style="font-size: 1.02rem; line-height: 1.7; max-width: 800px; opacity: 0.95; margin-bottom: 16px;">
            MoveInsight combina visión por computador y modelos de estimación de pose (YOLOv8-Pose) para
            extraer, cuadro a cuadro, los puntos clave del cuerpo humano en un video y transformarlos en
            métricas biomecánicas objetivas: ángulos articulares, simetría entre lado izquierdo y derecho,
            rango de movimiento y patrones de marcha.
        </p>
        <p style="font-size: 1.02rem; line-height: 1.7; max-width: 800px; opacity: 0.95; margin-bottom: 16px;">
            El sistema está pensado como una herramienta de apoyo para profesionales de la salud, entrenadores
            y personas interesadas en monitorear su movimiento a lo largo del tiempo, no como un dispositivo
            médico ni un sustituto de la evaluación clínica profesional.
        </p>
        <div style="display:flex; gap: 40px; margin-top: 28px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 1.8rem; font-weight: 800;">17</div>
                <div style="font-size: 0.85rem; opacity: 0.85;">Puntos clave detectados por cuadro</div>
            </div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 800;">9</div>
                <div style="font-size: 0.85rem; opacity: 0.85;">Tipos de análisis disponibles</div>
            </div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 800;">100%</div>
                <div style="font-size: 0.85rem; opacity: 0.85;">Procesamiento local, sin almacenamiento de video</div>
            </div>
        </div>
    </div>
    


    """
    st.markdown('<div id="sec-sobre-sistema"></div>', unsafe_allow_html=True)
    st.markdown(safe_html(about_html), unsafe_allow_html=True)

    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)

    

# ==============================================================================
# DASHBOARD VIEW
# ==============================================================================
else:
    with st.sidebar:
        st.markdown(safe_html(f'<div style="margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid #E2E8F0; display:flex; align-items:center;">{LOGO_HTML}</div>'), unsafe_allow_html=True)
        st.markdown(safe_html('<div class="section-title">1. CARGAR VIDEO</div>'), unsafe_allow_html=True)
        uploaded = st.file_uploader("Arrastra tu video aquí", type=["mp4","avi","mov"], label_visibility="collapsed")
        
        st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
        
        st.markdown(safe_html('<div class="section-title">2. SELECCIONAR ANÁLISIS</div>'), unsafe_allow_html=True)
        options = list(MODULES.keys())
        
        def_choice = 0
        if st.session_state.get("selected_choice") in options:
            def_choice = options.index(st.session_state["selected_choice"])
            
        choice = st.radio("Análisis", options, index=def_choice, format_func=lambda x: f"{x}", label_visibility="collapsed")

        st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
        analyze = st.button("INICIAR ANÁLISIS", disabled=(uploaded is None), type="primary", use_container_width=True)
        if st.button("Volver al Inicio", use_container_width=True):
            _reset()
            st.session_state["started"] = False
            st.rerun()

    dash_nav = """
    <div class="moveinsight-nav">
        <div></div>
        <div class="moveinsight-nav-links">
        </div>
    </div>
    """
    st.markdown(safe_html(dash_nav), unsafe_allow_html=True)

    if (analyze and uploaded is not None) or (st.session_state.get("uploaded_file") is not None and "video_path" not in st.session_state):
        run_file = uploaded if (analyze and uploaded is not None) else st.session_state.get("uploaded_file")
        run_choice = choice if (analyze and uploaded is not None) else st.session_state.get("selected_choice")
        
        from core.pose_extractor import process_video
        from reports.report_builder import build_summary_text, export_pdf, render_metric_expander

        module = MODULES[run_choice]
        suffix = Path(run_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(run_file.getbuffer())
            in_path = tmp.name

        out_dir = APP_DIR / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"ms_{int(time.time())}_{run_choice.replace(' ','_').replace('/','')}.mp4")

        st.markdown(safe_html('<div class="section-title">3. PROCESAMIENTO</div>'), unsafe_allow_html=True)
        col_pbar, col_status = st.columns([5, 1])
        with col_pbar:
            pbar = st.progress(0.0)
        with col_status:
            status_ph = st.empty()
            status_ph.markdown(safe_html('<div style="color: #64748B; font-weight: 600; font-size: 0.9rem; text-align:right;">Calculando...</div>'), unsafe_allow_html=True)

        def _cb(frac, fi, tot):
            pbar.progress(min(frac, 1.0))
            if fi % 15 == 0:
                status_ph.markdown(safe_html(f'<div style="color: #008B8B; font-weight: 700; font-size: 0.9rem; text-align:right;">{frac*100:.0f}%</div>'), unsafe_allow_html=True)

        try:
            with st.spinner(""):
                mhist, vinfo = process_video(
                    video_path=in_path, output_path=out_path,
                    module=module, alpha=0.4,
                    confidence_min=0.35, visibility_threshold=0.15,
                    progress_callback=_cb,
                )
            pbar.progress(1.0)
            status_ph.markdown(safe_html('<div style="color: #1E8E3E; font-weight: 700; font-size: 0.9rem; text-align:right;">Completado</div>'), unsafe_allow_html=True)

            summary      = module.compute_summary(mhist, vinfo["fps"], vinfo["duration_s"])
            thresholds   = module.THRESHOLDS
            summ_text    = build_summary_text(run_choice, summary, vinfo["duration_s"], thresholds=thresholds)
            logo_path    = str(APP_DIR / "assets" / "final_logo.png")
            pdf_bytes    = export_pdf(run_choice, summary, vinfo["duration_s"], logo_path=logo_path, thresholds=thresholds)

            st.session_state.update({
                "video_path":    out_path,
                "summary":       summary,
                "video_info":    vinfo,
                "analysis_name": run_choice,
                "summary_text":  summ_text,
                "pdf_bytes":     pdf_bytes,
            })
            st.session_state.pop("uploaded_file", None)
            st.session_state.pop("selected_choice", None)
            
            try: os.unlink(in_path)
            except Exception: pass
            st.rerun()
            
        except Exception as e:
            pbar.empty(); status_ph.empty()
            try: os.unlink(in_path)
            except Exception: pass
            st.error(f"❌ Error: {e}")
            st.exception(e)
            st.session_state.pop("uploaded_file", None)

    elif "video_path" in st.session_state:
        from reports.report_builder import render_metric_expander, metric_card_html, render_narrative_summary, build_report_data, build_narrative_summary
        vpath    = st.session_state["video_path"]
        summary  = st.session_state["summary"]
        aname    = st.session_state["analysis_name"]
        summ_txt = st.session_state["summary_text"]
        pdf_b    = st.session_state["pdf_bytes"]

        st.markdown(safe_html('<div class="section-title">3. PROCESAMIENTO</div>'), unsafe_allow_html=True)
        col_pbar, col_status = st.columns([5, 1])
        with col_pbar:
            st.progress(1.0)
        with col_status:
            st.markdown(safe_html('<div style="background: #E6F4EA; color: #1E8E3E; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; text-align: center; border: 1px solid #1E8E3E33; white-space:nowrap;">Análisis completado</div>'), unsafe_allow_html=True)
        
        st.markdown('<div style="height:32px; border-bottom: 1px solid #E2E8F0; margin-bottom: 32px;"></div>', unsafe_allow_html=True)

        col_vid, col_met = st.columns([1, 1], gap="large")
        with col_vid:
            st.markdown(safe_html('<div class="section-title">4. VIDEO ANALIZADO</div>'), unsafe_allow_html=True)
            
            # Mostrar video usando st.video() con BytesIO
            if vpath and os.path.exists(vpath):
                file_size = os.path.getsize(vpath)
                st.caption(f"📁 Archivo: {file_size / (1024*1024):.1f} MB")
                
                try:
                    with open(vpath, "rb") as f:
                        video_data = f.read()
                    
                    # Usar st.video() con BytesIO - forma estándar de Streamlit
                    st.video(io.BytesIO(video_data), format="video/mp4")
                    
                except Exception as e:
                    st.error(f"Error al cargar video: {e}")
            else:
                st.warning("⚠️ No se encontró el archivo de video")
            
            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
            
            # Botón de descarga
            if vpath and os.path.exists(vpath):
                with open(vpath, "rb") as f:
                    download_data = f.read()
                st.download_button("📥 Descargar Video (MP4)", data=download_data, file_name=f"moveinsight_{aname.replace(' ','_')}.mp4", mime="video/mp4", use_container_width=True)

        with col_met:
            st.markdown(safe_html(f'<div class="section-title">5. RESULTADOS - {aname.upper()}</div>'), unsafe_allow_html=True)
            if summary:
                order = {"alerta": 0, "atencion": 1, "normal": 2, "sin_datos": 3}
                sorted_m = sorted(summary.items(), key=lambda x: order.get(x[1].status, 9))
                for k, m in sorted_m:
                    st.markdown(metric_card_html(k, m), unsafe_allow_html=True)
                    render_metric_expander(aname, k, m, thresholds=MODULES[aname].THRESHOLDS)
            else:
                st.info("No se obtuvieron métricas suficientes.")
            st.download_button(
                "📄 Exportar Informe Clínico (PDF)",
                data=pdf_b,
                file_name=f"informe_{aname.replace(' ','_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)
        
        c3, c4 = st.columns([1.5, 1], gap="large")
        with c3:
            st.markdown(safe_html('<div class="section-title">6. RESUMEN DEL ANÁLISIS</div>'), unsafe_allow_html=True)
            with st.container(border=True):
                mod = MODULES[aname]
                report = build_report_data(
                    aname, summary, st.session_state["video_info"]["duration_s"],
                    thresholds=mod.THRESHOLDS,
                )
                sections = build_narrative_summary(report)
                render_narrative_summary(sections)
        with c4:
            st.markdown(safe_html('<div class="section-title">7. ACLARACIÓN IMPORTANTE</div>'), unsafe_allow_html=True)
            st.markdown(safe_html('<div style="background: #FFFBEB; border: 1px solid #FEF3C7; border-radius: 12px; padding: 24px; color: #92400E; font-size: 0.9rem; min-height: 120px;">Esta herramienta es un sistema de apoyo al análisis de movimiento. Los valores son orientativos y no constituyen diagnóstico médico.</div>'), unsafe_allow_html=True)
