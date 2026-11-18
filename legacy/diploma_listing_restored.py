"""Near-source reconstruction of the code listing on PDF pages 89-93.

NOT byte-identical to an original source file: the PDF lost indentation and
inserted page breaks. The original absolute path was deliberately removed.
Uses the historical Flet 0.28 API as shown in the dissertation.
"""

import base64
from io import BytesIO
from pathlib import Path

import flet as ft
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet


FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "IntensV2.xlsx"


def run_forecast(page, selected_km):
    try:
        while len(page.controls) > 1:
            page.controls.pop()
        page.update()

        data = pd.read_excel(FILE_PATH)
        page.add(ft.Text(f"Загружено строк: {len(data)}", color="blue"))
        page.add(ft.Text(f"Доступные колонки: {', '.join(data.columns)}", color="blue"))
        data.columns = data.columns.str.strip().str.replace(r"[\n ]", "", regex=True)
        if "Дата" not in data.columns:
            page.add(ft.Text("Ошибка: Столбец 'Дата' не найден после очистки!", color="red"))
            return
        if selected_km not in data.columns:
            page.add(ft.Text(f"Ошибка: Столбец {selected_km} не найден!", color="red"))
            return

        time_series = data[["Дата", selected_km]].copy()
        try:
            time_series["ds"] = pd.to_datetime(time_series["Дата"], errors="coerce")
            time_series["y"] = pd.to_numeric(time_series[selected_km], errors="coerce")
        except Exception as e:
            page.add(ft.Text(f"Ошибка преобразования данных: {str(e)}", color="red"))
            return
        time_series = time_series.dropna(subset=["ds", "y"])
        if len(time_series) == 0:
            page.add(ft.Text("Нет корректных данных для анализа!", color="red"))
            return
        page.add(ft.Text(f"Первые 5 строк данных:\n{time_series.head().to_string()}", color="blue"))

        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(time_series[["ds", "y"]])
        except Exception as e:
            page.add(ft.Text(f"Ошибка обучения модели: {str(e)}", color="red"))
            return

        future = model.make_future_dataframe(periods=365 * 5)
        forecast = model.predict(future)
        last_value = time_series["y"].iloc[-1]
        forecast_value = forecast["yhat"].iloc[-1]
        change_percent = ((forecast_value - last_value) / last_value) * 100 if last_value else float("nan")
        page.add(ft.Text(f"Последнее фактическое значение: {last_value:.2f} машин", size=16))
        page.add(ft.Text(f"Прогноз на 5 лет: {forecast_value:.2f} машин", size=18, weight="bold", color="green"))
        page.add(ft.Text(f"Изменение: {change_percent:.2f}%", color="green" if change_percent >= 0 else "red"))
        create_and_show_plots(page, model, forecast, time_series, selected_km)
    except Exception as e:
        page.add(ft.Text(f"Критическая ошибка: {str(e)}", color="red"))
    page.update()


def _encode_fig(fig):
    buffer = BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=100)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        plt.close(fig)
        buffer.close()


def create_and_show_plots(page, model, forecast, time_series, selected_km):
    try:
        fig1 = model.plot(forecast)
        fig1.suptitle(f"Прогноз интенсивности движения для {selected_km}")
        fig1.axes[0].set_xlabel("Дата")
        fig1.axes[0].set_ylabel("Интенсивность движения (машины)")
        encoded1 = _encode_fig(fig1)

        fig2 = model.plot_components(forecast)
        encoded2 = _encode_fig(fig2)

        fig3, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_series["ds"], time_series["y"], ".", label="Фактические данные")
        ax.plot(forecast["ds"], forecast["yhat"], "-", label="Прогноз")
        ax.fill_between(
            forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
            color="gray", alpha=0.2, label="Доверительный интервал"
        )
        ax.set_title(f"Фактические данные и прогноз для {selected_km}")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Интенсивность движения (машины)")
        ax.legend()
        encoded3 = _encode_fig(fig3)

        page.add(
            ft.Text("График прогноза:", size=16, weight="bold"),
            ft.Image(src_base64=encoded1, width=800, height=400),
            ft.Text("Компоненты прогноза:", size=16, weight="bold"),
            ft.Image(src_base64=encoded2, width=800, height=500),
            ft.Text("Сравнение фактических данных и прогноза:", size=16, weight="bold"),
            ft.Image(src_base64=encoded3, width=800, height=400),
        )
        page.update()
    except Exception as e:
        page.add(ft.Text(f"Ошибка при создании графиков: {str(e)}", color="red"))
        page.update()


def main(page: ft.Page):
    page.title = "Прогноз интенсивности движения"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20
    km_options = [
        "Интенсивностькм228",
        "Интенсивностькм322",
        "Интенсивностькм339",
        "Интенсивность416",
        "Интенсивностькм460",
    ]
    selected_km = ft.Dropdown(
        options=[ft.dropdown.Option(km) for km in km_options],
        value=km_options[0], width=300, label="Выберите километр", autofocus=True
    )
    forecast_button = ft.ElevatedButton(
        text="Выполнить прогноз", icon=ft.icons.TRENDING_UP,
        on_click=lambda _: run_forecast(page, selected_km.value), height=50, width=200
    )
    page.add(ft.Column([
        ft.Text("Прогноз интенсивности движения", size=24, weight="bold", color=ft.colors.BLUE_800),
        ft.Divider(height=20), selected_km, ft.Divider(height=10),
        forecast_button, ft.Divider(height=20)
    ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER))


if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)