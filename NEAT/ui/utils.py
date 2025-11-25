"""Shared UI utilities for NEAT."""

from PyQt5.QtWidgets import QApplication

def update_all_widget_fonts(new_font):
    for widget in QApplication.allWidgets():
        widget.setFont(new_font)
        widget.update()

__all__ = ["update_all_widget_fonts"]
