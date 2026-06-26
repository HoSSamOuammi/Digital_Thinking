from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "rapport_technique"
PDF_PATH = OUT_DIR / "rapport_technique.pdf"
MD_PATH = OUT_DIR / "rapport_technique.md"
ADMINS_DIR = ROOT / "static" / "Admins"

REPO_URL = "https://github.com/HoSSamOuammi/Digital_Thinking"
REPORT_DATE = date.today()
PROJECT_START = "04/03/2026"
TEAM_IMAGE_EXTENSIONS = (".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp")

TEAM = [
    {
        "slug": "aya",
        "name": "Aya EL Amrani",
        "email": "ElAamrani.aya@etu.uae.ac.ma",
        "role": "Architecture Flask, configuration, formulaires, stockage et routes.",
    },
    {
        "slug": "khadija",
        "name": "Khadija Baskar",
        "email": "Baskar.Khadija@etu.uae.ac.ma",
        "role": "Traduction francaise, libelles, contenus visibles et coherence UI.",
    },
    {
        "slug": "hossam",
        "name": "Hossam OUammi",
        "email": "Ouammi.hossam@etu.uae.ac.ma",
        "role": "Integration Flask, design administratif, medias, galerie et presentation.",
    },
    {
        "slug": "abdo",
        "name": "Abderrahmane El Garti",
        "email": "ElGarti.abderrahmane@etu.uae.ac.ma",
        "role": "Tests fonctionnels, documentation, rapport et analyse technique.",
    },
]

MODULES = [
    [
        "Tableau de bord",
        "Page d'accueil avec resume du studio, compteurs et acces rapides.",
        "Point d'entree clair pour presenter le projet.",
    ],
    [
        "Atelier generatif",
        "Generation de visuels a partir de series, palettes, graine, densite et accents dessines.",
        "Images exportees dans la galerie.",
    ],
    [
        "Donnees visuelles",
        "Lecture CSV ou donnees de demonstration, nettoyage numerique et transformation graphique.",
        "Visualisations en image.",
    ],
    [
        "Outils medias",
        "Traitement image: noir et blanc, sepia, contours, glitch, rotation, palette dominante. Audio optionnel.",
        "Fichiers transformes et telechargeables.",
    ],
    [
        "Galerie",
        "Listing separe des images et audios generes avec pagination.",
        "Consultation et telechargement des resultats.",
    ],
    [
        "Equipe",
        "Profils, roles, emails et photos chargees depuis static/Admins.",
        "Lien entre interface et repartition du travail.",
    ],
]

TOOLS = [
    ["Flask / Jinja2", "Routes serveur, templates HTML et formulaires."],
    ["Pillow", "Effets et exports d'images."],
    ["Pandas / NumPy", "Lecture, nettoyage et preparation des donnees CSV."],
    ["Matplotlib", "Production des visualisations de donnees."],
    ["PyDub / ffmpeg", "Traitement audio lorsque l'environnement le permet."],
    ["unittest", "Verification automatique des routes et parcours principaux."],
    ["Git / GitHub", "Historique, collaboration et depot final public."],
]

PIPELINE = [
    "L'utilisateur choisit une page atelier depuis le tableau de bord.",
    "Flask recoit le formulaire et valide les donnees utiles.",
    "Le module Python specialise genere ou transforme le contenu.",
    "Le fichier final est sauvegarde dans static/generated.",
    "La page affiche le resultat et propose le telechargement.",
    "Les tests verifient les routes, la securite CSRF, la galerie et le nettoyage des fichiers.",
]

CHALLENGES = [
    [
        "Code initial trop centralise",
        "Separation en create_app, routes, formulaires, stockage, securite et modules metier.",
    ],
    [
        "Interface a rendre presentable en contexte scolaire",
        "Design sobre, navigation simple, libelles francais et mise en page responsive.",
    ],
    [
        "Dependance audio sensible a l'environnement",
        "Detection de disponibilite et degradation propre si ffmpeg n'est pas installe.",
    ],
    [
        "Fichiers generes et imports temporaires",
        "Nettoyage automatique, pagination et limites de cache pour garder le depot propre.",
    ],
]

DELIVERABLES = [
    ["Base de code complete", "Dossiers app.py, studio, modules, templates, static, tests et requirements.txt."],
    ["Application Flask", "Lancement par python app.py, routes principales testees."],
    ["README", "Installation, lancement, structure, tests et notes techniques."],
    ["Rapport final", "PDF de 2-3 pages avec concept, modules, pipeline, challenges et solutions."],
]


def format_report_date() -> str:
    return REPORT_DATE.strftime("%d/%m/%Y")


def team_photo_path(member: dict[str, str]) -> Path | None:
    for extension in TEAM_IMAGE_EXTENSIONS:
        candidate = ADMINS_DIR / f"{member['slug']}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def markdown_photo(member: dict[str, str]) -> str:
    photo_path = team_photo_path(member)
    if not photo_path:
        return "-"
    return f"![{member['name']}](../../static/Admins/{photo_path.name})"


def build_markdown() -> None:
    lines = [
        "# Rapport final - Studio generatif interactif",
        "",
        f"- Date de verification : {format_report_date()}",
        f"- Periode de travail : {PROJECT_START} au {format_report_date()}",
        f"- Depot GitHub : {REPO_URL}",
        "- Objet : application Flask de creativite numerique.",
        "",
        "## 1. Concept et direction artistique",
        "",
        "Le projet est un studio generatif interactif qui rassemble plusieurs ateliers numeriques dans une interface unique. "
        "L'objectif artistique est de transformer des parametres simples, des donnees et des medias en productions visuelles "
        "exportables, tout en gardant une experience claire pour une presentation scolaire.",
        "",
        "La direction visuelle choisie est sobre et administrative : fond clair, cartes lisibles, boutons simples, palette "
        "bleu-vert avec accents limites et textes francais. Ce choix met en avant le fonctionnement de l'application plus que "
        "la decoration.",
        "",
        "## 2. Modules implementes",
        "",
        "| Module | Fonction | Resultat |",
        "| --- | --- | --- |",
    ]
    for module, function, result in MODULES:
        lines.append(f"| {module} | {function} | {result} |")

    lines.extend(
        [
            "",
            "## 3. Outils utilises et pipeline technique",
            "",
            "| Outil | Utilisation |",
            "| --- | --- |",
        ]
    )
    for tool, usage in TOOLS:
        lines.append(f"| {tool} | {usage} |")

    lines.append("")
    lines.append("Pipeline :")
    for step in PIPELINE:
        lines.append(f"- {step}")

    lines.extend(
        [
            "",
            "## 4. Challenges et solutions",
            "",
            "| Challenge | Solution |",
            "| --- | --- |",
        ]
    )
    for challenge, solution in CHALLENGES:
        lines.append(f"| {challenge} | {solution} |")

    lines.extend(
        [
            "",
            "## 5. Equipe et livrables",
            "",
            "| Photo | Membre | Role |",
            "| --- | --- | --- |",
        ]
    )
    for member in TEAM:
        lines.append(f"| {markdown_photo(member)} | {member['name']} | {member['role']} |")

    lines.extend(["", "| Livrable | Verification |", "| --- | --- |"])
    for deliverable, verification in DELIVERABLES:
        lines.append(f"| {deliverable} | {verification} |")

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#202124"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=colors.HexColor("#202124"),
            spaceBefore=7,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.2,
            textColor=colors.HexColor("#202124"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHead",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=8.4,
            textColor=colors.HexColor("#202124"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.3,
            textColor=colors.HexColor("#202124"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9.2,
            textColor=colors.HexColor("#3c4043"),
        )
    )
    return styles


STYLES = make_styles()


def paragraph(text: str, style: str = "BodySmall") -> Paragraph:
    return Paragraph(escape(text), STYLES[style])


def table_cell(value, *, header: bool = False):
    if hasattr(value, "wrap"):
        return value
    return Paragraph(escape(str(value)), STYLES["TableHead" if header else "TableText"])


def styled_table(data, widths):
    rows = []
    for row_index, row in enumerate(data):
        rows.append([table_cell(value, header=row_index == 0) for value in row])

    output = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    output.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd7e3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return output


def bullet_list(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(paragraph(item, "TableText"), leftIndent=8) for item in items],
        bulletType="bullet",
        leftIndent=12,
        bulletFontSize=5,
    )


def member_photo(member: dict[str, str]):
    photo_path = team_photo_path(member)
    if not photo_path:
        return paragraph("-", "TableText")

    image = Image(str(photo_path))
    max_width = 1.15 * cm
    max_height = 1.45 * cm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def build_pdf() -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.25 * cm,
        leftMargin=1.25 * cm,
        topMargin=1.15 * cm,
        bottomMargin=1.1 * cm,
        title="Rapport final - Studio generatif interactif",
    )
    story = []

    story.append(Paragraph("Rapport final", STYLES["ReportTitle"]))
    story.append(paragraph("Studio generatif interactif - Application Flask de creativite numerique"))
    story.append(
        styled_table(
            [
                ["Element", "Information"],
                ["Date de verification", format_report_date()],
                ["Periode de travail", f"{PROJECT_START} au {format_report_date()}"],
                ["Depot GitHub", REPO_URL],
                ["Objet du livrable", "Code complet, application Flask, README et rapport final 2-3 pages."],
            ],
            [4.0 * cm, 13.6 * cm],
        )
    )
    story.append(Spacer(1, 0.18 * cm))

    story.append(Paragraph("1. Concept et direction artistique", STYLES["Section"]))
    story.append(
        paragraph(
            "Le projet est un studio generatif interactif qui regroupe des ateliers de generation visuelle, "
            "visualisation de donnees, traitement media et galerie. L'utilisateur transforme des parametres, "
            "des fichiers CSV ou des images en productions exportables."
        )
    )
    story.append(Spacer(1, 0.08 * cm))
    story.append(
        paragraph(
            "La direction artistique reste sobre: interface claire, composants administratifs, couleurs limitees "
            "et textes francais. Le rendu privilegie la lisibilite et la demonstration technique."
        )
    )

    story.append(Paragraph("2. Modules implementes", STYLES["Section"]))
    story.append(styled_table([["Module", "Fonction", "Resultat"], *MODULES], [3.2 * cm, 8.1 * cm, 6.3 * cm]))

    story.append(PageBreak())
    story.append(Paragraph("3. Outils utilises et pipeline technique", STYLES["Section"]))
    story.append(styled_table([["Outil", "Utilisation"], *TOOLS], [4.2 * cm, 13.4 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(paragraph("Pipeline de fonctionnement:"))
    story.append(bullet_list(PIPELINE))

    story.append(PageBreak())
    story.append(Paragraph("4. Challenges et solutions", STYLES["Section"]))
    story.append(styled_table([["Challenge", "Solution"], *CHALLENGES], [5.7 * cm, 11.9 * cm]))

    story.append(Paragraph("5. Equipe et livrables verifies", STYLES["Section"]))
    story.append(
        styled_table(
            [["Photo", "Membre", "Role"]]
            + [[member_photo(member), member["name"], member["role"]] for member in TEAM],
            [1.55 * cm, 4.0 * cm, 12.05 * cm],
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(styled_table([["Livrable", "Verification"], *DELIVERABLES], [4.5 * cm, 13.1 * cm]))

    doc.build(story)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_markdown()
    build_pdf()
    print(PDF_PATH)
    print(MD_PATH)
