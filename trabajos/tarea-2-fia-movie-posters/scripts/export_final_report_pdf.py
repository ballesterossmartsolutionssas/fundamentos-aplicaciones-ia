from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "execution" / "final-colab-run"
REPORTS_DIR = ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "informe_tarea_2_fia_movie_posters.pdf"

UNIVERSITY = "Universidad Autonoma de Occidente"
FACULTY = "Facultad de Ingenieria"
COURSE = "Fundamentos y Aplicaciones de Inteligencia Artificial"
TITLE = "Tarea 2 FIA"
SUBTITLE = "Clasificacion multi-label de generos de peliculas usando posters"
PROFESSOR = "Juan Sebastian Mosquera Maturana"
AUTHORS = [
    "Valentina Popo Montilla",
    "Juan Camilo Balleresteros Sierra",
    "Santigo Rodriguez Gacha",
]

BLUE = colors.HexColor("#153B64")
SKY = colors.HexColor("#DCEAF7")
PALE = colors.HexColor("#F5F8FC")
INK = colors.HexColor("#20242A")
MUTED = colors.HexColor("#5B6472")
GREEN = colors.HexColor("#2E7D50")


def money(value: float) -> str:
    return f"{value:.4f}"


def pct(value: float) -> str:
    return f"{value:.2%}"


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(RESULTS_DIR / "training_history.csv")
    roc = pd.read_csv(RESULTS_DIR / "roc_auc_scores.csv").sort_values(
        "roc_auc", ascending=False
    )
    report = pd.read_csv(RESULTS_DIR / "classification_report.csv", index_col=0)
    return history, roc, report


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverKicker",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1Clean",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyClean",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallMuted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricValue",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=GREEN,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricLabel",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
        )
    )
    return styles


def paragraph(text: str, styles):
    return Paragraph(text, styles["BodyClean"])


def simple_table(rows, col_widths=None):
    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SKY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.6),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB4C0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def metric_cards(metrics, styles):
    cells = []
    for label, value in metrics:
        cells.append(
            Table(
                [
                    [Paragraph(value, styles["MetricValue"])],
                    [Paragraph(label, styles["MetricLabel"])],
                ],
                colWidths=[3.75 * cm],
            )
        )
    table = Table(
        [cells],
        colWidths=[4.05 * cm] * len(cells),
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CAD5E2")),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#CAD5E2")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def add_cover(story, styles):
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph(UNIVERSITY, styles["CoverKicker"]))
    story.append(Paragraph(FACULTY, styles["CoverKicker"]))
    story.append(Paragraph(COURSE, styles["CoverKicker"]))
    story.append(Spacer(1, 1.1 * cm))
    story.append(Paragraph(TITLE, styles["CoverTitle"]))
    story.append(Paragraph(SUBTITLE, styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.3 * cm))
    info_rows = [
        ["Tipo de tarea", "Clasificacion multi-label con CNN"],
        ["Dataset", "Movies-Poster_Dataset"],
        ["Entorno final", "Google Colab - GPU T4"],
        ["Fecha", datetime.now().strftime("%d/%m/%Y")],
    ]
    story.append(simple_table(info_rows, [5 * cm, 9 * cm]))
    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph(f"<b>Profesor:</b> {PROFESSOR}", styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>Integrantes</b>", styles["CoverSubtitle"]))
    for author in AUTHORS:
        story.append(Paragraph(author, styles["CoverSubtitle"]))
    story.append(PageBreak())


def add_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(BLUE)
    canvas.rect(0, height - 1.0 * cm, width, 1.0 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawString(2 * cm, height - 0.62 * cm, "Tarea 2 FIA - Movie Poster Classification")
    canvas.setFont("Helvetica", 8.2)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 2 * cm, 1.1 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def build_story():
    history, roc, report = load_results()
    styles = build_styles()
    final = history.iloc[-1]
    story = []
    add_cover(story, styles)

    story.append(Paragraph("Resumen ejecutivo", styles["H1Clean"]))
    story.append(
        paragraph(
            "Se implemento una red neuronal convolucional original para clasificar generos de peliculas a partir de posters. "
            "El problema se formulo como clasificacion multi-label, usando activacion sigmoid y perdida binary_crossentropy, "
            "porque una misma pelicula puede pertenecer simultaneamente a varios generos.",
            styles,
        )
    )
    story.append(
        metric_cards(
            [
                ("val_auc", money(final["val_auc"])),
                ("val_binary_accuracy", money(final["val_binary_accuracy"])),
                ("val_loss", money(final["val_loss"])),
                ("epocas", str(len(history))),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Datos y procesamiento", styles["H1Clean"]))
    story.append(
        paragraph(
            "El notebook clona el dataset desde GitHub, lee el CSV principal, asocia cada registro con su imagen por medio del identificador Id/IMDB y transforma los generos en etiquetas binarias. "
            "La lectura de imagenes se realiza con ImageDataGenerator.flow_from_dataframe, cumpliendo la recomendacion del enunciado.",
            styles,
        )
    )
    story.append(
        simple_table(
            [
                ["Paso", "Descripcion"],
                ["1", "Clonacion del repositorio Movies-Poster_Dataset."],
                ["2", "Lectura del CSV y deteccion de columnas Id y Genre."],
                ["3", "Cruce entre Id de pelicula y archivo de imagen."],
                ["4", "Conversion de generos a etiquetas multi-label con MultiLabelBinarizer."],
                ["5", "Division en entrenamiento, validacion y prueba."],
                ["6", "Aumento de datos y normalizacion con ImageDataGenerator."],
            ],
            [1.4 * cm, 14.5 * cm],
        )
    )

    story.append(Paragraph("Modelo propuesto", styles["H1Clean"]))
    story.append(
        paragraph(
            "La CNN usa bloques Conv2D, BatchNormalization y MaxPooling2D para extraer patrones visuales, seguidos por GlobalAveragePooling2D, Dropout y capas densas. "
            "La capa final tiene una neurona por genero y activacion sigmoid. Este diseno mantiene la arquitectura simple, original y adecuada para la sustentacion.",
            styles,
        )
    )
    story.append(
        simple_table(
            [
                ["Componente", "Funcion"],
                ["Conv2D", "Extrae bordes, texturas, formas y composicion visual del poster."],
                ["BatchNormalization", "Estabiliza el entrenamiento."],
                ["MaxPooling2D", "Reduce resolucion espacial y costo computacional."],
                ["Dropout", "Reduce sobreajuste."],
                ["Sigmoid", "Permite activar varios generos al mismo tiempo."],
                ["Binary crossentropy", "Evalua cada etiqueta como una decision binaria independiente."],
            ],
            [4.2 * cm, 11.7 * cm],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Resultados finales", styles["H1Clean"]))
    story.append(
        paragraph(
            "La ejecucion final se realizo en Google Colab con GPU T4 durante 10 epocas. "
            "La mejor metrica de validacion se obtuvo al cierre del entrenamiento, con val_auc de 0.6830 y val_loss de 0.2919.",
            styles,
        )
    )
    story.append(
        simple_table(
            [
                ["Metrica", "Valor"],
                ["binary_accuracy entrenamiento", money(final["binary_accuracy"])],
                ["auc entrenamiento", money(final["auc"])],
                ["loss entrenamiento", money(final["loss"])],
                ["val_binary_accuracy", money(final["val_binary_accuracy"])],
                ["val_auc", money(final["val_auc"])],
                ["val_loss", money(final["val_loss"])],
                ["ROC-AUC promedio por genero", money(roc["roc_auc"].mean())],
                ["ROC-AUC mediano por genero", money(roc["roc_auc"].median())],
                ["F1 micro", money(report.loc["micro avg", "f1-score"])],
                ["F1 ponderado", money(report.loc["weighted avg", "f1-score"])],
                ["F1 por muestras", money(report.loc["samples avg", "f1-score"])],
            ],
            [7 * cm, 5 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    chart = Image(str(RESULTS_DIR / "training_curves.png"))
    chart._restrictSize(16.2 * cm, 7.8 * cm)
    story.append(KeepTogether([chart, Paragraph("Figura 1. Curvas de accuracy y loss.", styles["SmallMuted"])]))

    story.append(PageBreak())
    story.append(Paragraph("ROC-AUC por genero", styles["H1Clean"]))
    story.append(
        paragraph(
            "El ROC-AUC por genero muestra mejor separacion en categorias visualmente mas distinguibles y con patrones recurrentes en los posters. "
            "Los generos minoritarios o mas ambiguos presentan menor rendimiento.",
            styles,
        )
    )
    top_rows = [["Genero", "ROC-AUC"]] + [
        [row.genre, money(row.roc_auc)] for row in roc.head(8).itertuples(index=False)
    ]
    low_rows = [["Genero", "ROC-AUC"]] + [
        [row.genre, money(row.roc_auc)] for row in roc.tail(6).itertuples(index=False)
    ]
    story.append(
        Table(
            [[simple_table(top_rows, [5 * cm, 3 * cm]), simple_table(low_rows, [5 * cm, 3 * cm])]],
            colWidths=[8.4 * cm, 8.4 * cm],
        )
    )
    story.append(Spacer(1, 0.35 * cm))
    roc_img = Image(str(RESULTS_DIR / "roc_auc_by_genre.png"))
    roc_img._restrictSize(15.5 * cm, 11.3 * cm)
    story.append(KeepTogether([roc_img, Paragraph("Figura 2. Curvas ROC por genero.", styles["SmallMuted"])]))

    story.append(PageBreak())
    story.append(Paragraph("Interpretacion", styles["H1Clean"]))
    story.append(
        paragraph(
            "La binary_accuracy es alta porque en una tarea multi-label existen muchas etiquetas negativas por cada imagen. "
            "Por esta razon, la interpretacion debe apoyarse tambien en el ROC-AUC por genero y en la prueba top 5 de una caratula externa. "
            "El classification report evidencia que el umbral fijo de 0.35 favorece generos frecuentes como Drama y Comedy, mientras que generos con menos muestras tienen menor recall.",
            styles,
        )
    )
    story.append(
        simple_table(
            [
                ["Hallazgo", "Lectura"],
                ["Buen desempeno global", "val_auc de 0.6830 y ROC-AUC promedio de 0.6756."],
                ["Generos destacados", "Animation, Comedy, Horror, Family y Sci-Fi superan 0.75 de ROC-AUC."],
                ["Limitacion principal", "Desbalance de clases y bajo recall en generos minoritarios."],
                ["Mejora futura", "Ajustar umbrales por genero, aplicar balanceo o entrenar mas tiempo."],
            ],
            [5 * cm, 11 * cm],
        )
    )

    story.append(Paragraph("Prueba con caratula externa", styles["H1Clean"]))
    story.append(
        paragraph(
            "El notebook incluye una celda para subir una imagen externa con files.upload(), redimensionarla al tamano usado por la red, normalizarla y mostrar los cinco generos con mayor probabilidad. "
            "Esta prueba es la parte practica que permite contrastar la prediccion del modelo con los generos reales consultados en IMDB.",
            styles,
        )
    )

    story.append(Paragraph("Conclusiones", styles["H1Clean"]))
    story.append(
        paragraph(
            "La solucion cumple el objetivo del enunciado: implementa lectura de imagenes desde dataframe, una CNN original, evaluacion multi-label y prueba de una caratula externa. "
            "Los resultados son razonables para una CNN entrenada desde cero sobre un dataset desbalanceado. Para la sustentacion, conviene resaltar la diferencia entre clasificacion multiclase y multi-label, el uso de sigmoid y binary_crossentropy, y la interpretacion cuidadosa de binary_accuracy frente a ROC-AUC.",
            styles,
        )
    )

    return story


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=1.75 * cm,
        leftMargin=1.75 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.8 * cm,
    )
    doc.build(build_story(), onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(f"PDF generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
