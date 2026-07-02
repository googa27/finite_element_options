"""Streamlit demo instantiating provided problem presets."""

import streamlit as st

from finite_element_options.problems import CreditRiskProblem, OptionPricingProblem


def main() -> None:
    """Render the demo page."""

    st.title("Problem Presets")
    st.subheader("Option Pricing")
    st.write(OptionPricingProblem())
    st.subheader("Credit Risk")
    st.write(CreditRiskProblem())


if __name__ == "__main__":
    main()
