"""Traffic intensity forecasting interface reconstructed from the 2025 thesis.

The thesis appendix (PDF pp. 89-93) includes a Flet application based on
Prophet. This version repairs formatting and enables running on a demo CSV,
without distributing the private/third-party original Excel workbook.
"""

import base64
import os
from io import BytesIO
from pathlib import Path

import flet as ft
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "sample_traffic.csv"
# Original thesis column names are preserved, including "Интенсивность416".
KM_OPTIONS = [
    "Интенсивностькм228",
    "Интенсивностькм322",
    "Интенсивностькм339",
    "Интенсивность416",
    "Интенсивностькм460",
]


def load_time_series(file_path: Path, selected_km: str) -> pd.DataFrame:
    """Return Prophet-ready ds/y data for one road segment.

    Supports Excel (.xlsx, .xls) and CSV. Raises informative exceptions for
    missing files, columns, or insufficient valid observations.
    """
    file_path = Path(file_path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл данных не найден: {file_path}")

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(file_path)
    elif file_path.suffix.lower() == ".csv":
        data = pd.read_csv(file_path, encoding="utf-8-sig")
    else:
        raise ValueError("Поддерживаются только CSV и Excel (.xlsx/.xls).")

    # In the PDF, spaces/newlines were removed from names of columns.
    data.columns = data.columns.astype(str).str.strip().str.replace(r"[\n ]", "", regex=True)
    if "Дата" not in data.columns:
        raise ValueError("Не найден столбец 'Дата'.")
    if selected_km not in data.columns:
        raise ValueError(
            f"Не найден столбец '{selected_km}'. "
            f"Доступные столбцы: {', '.join(data.columns)}"
        )

    time_series = data[["Дата", selected_km]].copy()
    # The demo uses ISO dates (YYYY-MM-DD); Excel exports may use DD.MM.YYYY.
    sample_dates = time_series["Дата"].dropna().astype(str)
    if not sample_dates.empty and sample_dates.iloc[0][:4].isdigit() and sample_dates.iloc[0][4:5] == "-":
        time_series["ds"] = pd.to_datetime(time_series["Дата"], errors="coerce", format="%Y-%m-%d")
    else:
        time_series["ds"] = pd.to_datetime(time_series["Дата"], errors="coerce", dayfirst=True)
    time_series["y"] = pd.to_numeric(time_series[selected_km], errors="coerce")
    time_series = time_series.dropna(subset=["ds", "y"])[["ds", "y"]]
    time_series = time_series.sort_values("ds")
    # Coalesce any duplicate dates to one daily observation.
    time_series = time_series.groupby("ds", as_index=False)["y"].mean()
    if len(time_series) < 10:
        raise ValueError("Для прогноза нужно не менее 10 корректных наблюдений.")
    return time_series


def make_forecast(time_series: pd.DataFrame):
    """Fit the same Prophet model/settings documented in the thesis."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(time_series[["ds", "y"]])
    # The appendix forecasts 365 * 5 daily observations into the future.
    future = model.make_future_dataframe(periods=365 * 5, freq="D")
    forecast = model.predict(future)
    return model, forecast


def _figure_as_flet_image(fig, width=800, height=400):
    buffer = BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return ft.Image(src_base64=encoded, width=width, height=height, fit=ft.ImageFit.CONTAIN)
    finally:
        buffer.close()
        plt.close(fig)


def create_and_show_plots(results: ft.Column, model, forecast, time_series, selected_km):
    """Visualizations from pp. 91-92: forecast, components, actual vs forecast."""
    first_figure = model.plot(forecast, figsize=(12, 6))
    first_figure.suptitle(f"Прогноз интенсивности движения: {selected_km}")
    first_figure.axes[0].set_ylabel("Интенсивность движения (машины)")

    second_figure = model.plot_components(forecast, figsize=(12, 8))

    third_figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(time_series["ds"], time_series["y"], ".", label="Фактические данные")
    axis.plot(forecast["ds"], forecast["yhat"], "-", label="Прогноз")
    axis.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="gray",
        alpha=0.2,
        label="Интервал неопределённости",
    )
    axis.set_title(f"Фактические данные и прогноз: {selected_km}")
    axis.set_xlabel("Дата")
    axis.set_ylabel("Интенсивность движения (машины)")
    axis.legend()
    third_figure.autofmt_xdate()

    results.controls.extend(
        [
            ft.Text("Прогноз с интервалом неопределённости", size=16, weight="bold"),
            _figure_as_flet_image(first_figure),
            ft.Text("Компоненты прогноза", size=16, weight="bold"),
            _figure_as_flet_image(second_figure, height=500),
            ft.Text("Сравнение фактических данных и прогноза", size=16, weight="bold"),
            _figure_as_flet_image(third_figure),
        ]
    )


def main(page: ft.Page):
    page.title = "Прогноз интенсивности движения"
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    data_path = Path(os.getenv("TRAFFIC_DATA_FILE", str(DEFAULT_DATA_FILE)))
    results = ft.Column(spacing=12, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    selected_km = ft.Dropdown(
        options=[ft.dropdown.Option(km) for km in KM_OPTIONS],
        value=KM_OPTIONS[0],
        width=320,
        label="Выберите километр",
    )

    def run_forecast(_):
        results.controls.clear()
        button.disabled = True
        results.controls.append(ft.Text("Обработка данных и построение прогноза..."))
        page.update()
        try:
            time_series = load_time_series(data_path, selected_km.value)
            model, forecast = make_forecast(time_series)
            last_value = float(time_series["y"].iloc[-1])
            forecast_value = float(forecast["yhat"].iloc[-1])
            results.controls.clear()
            results.controls.append(ft.Text(f"Корректных наблюдений: {len(time_series)}"))
            results.controls.append(ft.Text(f"Последнее значение: {last_value:,.2f} машин", size=16))
            results.controls.append(
                ft.Text(f"Прогноз примерно на 5 лет вперёд: {forecast_value:,.2f} машин", size=18, weight="bold")
            )
            if last_value != 0:
                change_percent = (forecast_value - last_value) / last_value * 100
                results.controls.append(ft.Text(f"Изменение: {change_percent:+.2f}%"))
            else:
                results.controls.append(ft.Text("Изменение в процентах не вычисляется: исходное значение равно 0."))
            create_and_show_plots(results, model, forecast, time_series, selected_km.value)
        except Exception as exc:
            results.controls.clear()
            results.controls.append(ft.Text(f"Ошибка: {exc}", color="red"))
        finally:
            button.disabled = False
            page.update()

    button = ft.ElevatedButton(
        text="Выполнить прогноз",
        icon=ft.icons.TRENDING_UP,
        on_click=run_forecast,
        height=50,
        width=220,
    )
    page.add(
        ft.Column(
            controls=[
                ft.Text("Прогноз интенсивности движения", size=24, weight="bold", color=ft.colors.BLUE_800),
                ft.Text("Демонстрационный набор данных — синтетический; не реальные измерения!"),
                ft.Text(f"Источник данных: {data_path.name}", size=12),
                ft.Divider(),
                selected_km,
                button,
                ft.Divider(),
                results,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        )
    )


if __name__ == "__main__":
    # Flet 0.28.3 (the compatible generation for the thesis-era API).
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)