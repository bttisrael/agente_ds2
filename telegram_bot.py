import os
import pickle
import logging
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
import anthropic

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

df = None
model = None

AWAITING_FEATURE_1, AWAITING_FEATURE_2, AWAITING_FEATURE_3 = range(3)

TOP_FEATURES = [
    ('feat_ratio', 0.9997),
    ('days_for_shipment_scheduled', 0.0001),
    ('shipping_efficiency', 0.0001),
]

FEATURE_DESCRIPTIONS = {
    'feat_ratio': 'Ratio of actual to scheduled shipping days - critical for identifying delays',
    'days_for_shipment_scheduled': 'Planned shipping duration - shorter windows increase risk',
    'shipping_efficiency': 'Overall shipping performance metric - lower values indicate problems',
}


def load_data():
    global df, model
    try:
        df = pd.read_parquet('df4_predictions.parquet')
        logger.info(f"Loaded dataframe with {len(df)} rows")
        with open('final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Loaded model successfully")
    except Exception as e:
        logger.error(f"Error loading data or model: {e}")
        raise


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        welcome_message = """
🚀 **Welcome to the Supply Chain Risk Assistant!**

I help predict late delivery risks for DataCo Global's supply chain operations.

📊 **What I can do:**
• Analyze 180k+ orders to predict delivery risks
• Identify key factors causing delays
• Provide actionable insights for operations managers

**Available Commands:**

/stats - View dataset and model performance summary
/top_features - See most important risk factors
/hypotheses - Review validated business insights
/predict - Predict late delivery risk for new orders
/insights - Get AI-powered business insights
/help - Show all commands

Let's optimize your supply chain! 📦✨
"""
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in start: {e}")
        await update.message.reply_text("Sorry, an error occurred. Please try again.")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        total_records = len(df)
        
        pred_col = 'predictions' if 'predictions' in df.columns else 'late_delivery_risk'
        
        if pred_col in df.columns:
            predictions = df[pred_col].value_counts()
            class_0 = predictions.get(0.0, 0)
            class_1 = predictions.get(1.0, 0)
            pct_0 = (class_0 / total_records * 100) if total_records > 0 else 0
            pct_1 = (class_1 / total_records * 100) if total_records > 0 else 0
        else:
            class_0 = 77119
            class_1 = 103400
            pct_0 = 42.72
            pct_1 = 57.28
        
        avg_confidence = "N/A"
        if 'prediction_proba' in df.columns:
            avg_confidence = f"{df['prediction_proba'].mean():.4f}"
        
        stats_message = f"""
📊 **Dataset & Model Statistics**

**Dataset Overview:**
📦 Total Records: {total_records:,}
🤖 Model: XGBoost Classifier
🎯 Accuracy: 97.45%

**Prediction Distribution:**
✅ On-Time Deliveries (0): {class_0:,} ({pct_0:.2f}%)
⚠️ Late Deliveries (1): {class_1:,} ({pct_1:.2f}%)

**Model Confidence:**
📈 Average Confidence: {avg_confidence}

**Business Impact:**
Over 57% of orders are at risk of late delivery, highlighting critical need for proactive intervention and resource optimization.
"""
        await update.message.reply_text(stats_message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in stats: {e}")
        await update.message.reply_text("Sorry, couldn't retrieve statistics. Please try again.")


async def top_features(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        features_message = """
🔍 **Top 7 Most Important Risk Factors**

**1. feat_ratio (99.97%)**
   📊 Ratio of actual vs scheduled shipping time
   💼 Business Impact: The single most critical predictor - when actual shipping exceeds scheduled time, late delivery is almost certain

**2. days_for_shipment_scheduled (0.01%)**
   ⏱️ Planned shipping duration in days
   💼 Business Impact: Tighter delivery windows increase pressure and risk - helps identify aggressive scheduling

**3. shipping_efficiency (0.01%)**
   ⚡ Overall shipping performance metric
   💼 Business Impact: Measures carrier and route effectiveness - low efficiency signals systemic problems

**4. feat_interact (0.00%)**
   🔗 Interaction between shipping features
   💼 Business Impact: Captures complex relationships between timing factors

**5. days_for_shipping_real (0.00%)**
   📅 Actual shipping time taken
   💼 Business Impact: Ground truth of delivery performance

**6. shipping_time_interaction (0.00%)**
   ⚙️ Combined timing effects
   💼 Business Impact: Identifies compounding delays

**7. feat_sum (0.00%)**
   ➕ Aggregate shipping metrics
   💼 Business Impact: Overall delivery complexity indicator

**Key Takeaway:** Focus on feat_ratio monitoring - it accounts for 99.97% of predictive power!
"""
        await update.message.reply_text(features_message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in top_features: {e}")
        await update.message.reply_text("Sorry, couldn't retrieve feature importance. Please try again.")


async def hypotheses(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        hypotheses_message = """
✅ **Validated Business Hypotheses**

All hypotheses were confirmed as **TRUE** through statistical analysis:

**1. Tight Scheduling Increases Risk**
Orders with lower scheduled shipping days tend to have higher late delivery risk.
📌 Action: Build buffer time into tight delivery windows, especially for complex routes.

**2. Schedule Overruns Are Critical**
Orders where actual shipping days exceed scheduled days have significantly higher late delivery rates.
📌 Action: Implement real-time monitoring to catch delays early and trigger expedited handling.

**3. Transaction Type Matters**
Specific transaction types (DEBIT, TRANSFER, etc.) correlate with different risk levels.
📌 Action: Apply transaction-specific routing rules and resource allocation.

**4. Product Category Risk Patterns**
Certain product categories consistently show higher late delivery rates.
📌 Action: Pre-allocate extra handling resources for high-risk categories.

**5. Department Performance Varies**
Orders from specific departments have distinct delivery performance profiles.
📌 Action: Conduct targeted training and process improvements in underperforming departments.

**Strategic Recommendation:**
Implement a multi-factor risk scoring system based on these validated patterns to prioritize interventions where they'll have maximum impact.
"""
        await update.message.reply_text(hypotheses_message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in hypotheses: {e}")
        await update.message.reply_text("Sorry, couldn't retrieve hypotheses. Please try again.")


async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data.clear()
        
        message = """
🔮 **Late Delivery Risk Prediction**

I'll ask you for values of the top 3 features to predict delivery risk.

**Feature 1: feat_ratio**
This is the ratio of actual to scheduled shipping days.
- Value < 1.0: Shipping faster than scheduled (good)
- Value = 1.0: On schedule
- Value > 1.0: Behind schedule (risk!)

Please enter the **feat_ratio** value (e.g., 0.95, 1.0, 1.25):
"""
        await update.message.reply_text(message, parse_mode='Markdown')
        return AWAITING_FEATURE_1
    except Exception as e:
        logger.error(f"Error in predict_start: {e}")
        await update.message.reply_text("Sorry, couldn't start prediction. Please try again.")
        return ConversationHandler.END


async def predict_feature_1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        try:
            value = float(user_input)
            context.user_data['feat_ratio'] = value
            
            message = """
✅ Got it!

**Feature 2: days_for_shipment_scheduled**
This is the planned shipping duration in days (typically 2-5 days).
- Lower values = tighter timeline = higher risk
- Higher values = more buffer = lower risk

Please enter **days_for_shipment_scheduled** (e.g., 2, 3, 4):
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            return AWAITING_FEATURE_2
        except ValueError:
            await update.message.reply_text("Please enter a valid number (e.g., 1.25)")
            return AWAITING_FEATURE_1
    except Exception as e:
        logger.error(f"Error in predict_feature_1: {e}")
        await update.message.reply_text("Error processing input. Please try again.")
        return ConversationHandler.END


async def predict_feature_2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        try:
            value = float(user_input)
            context.user_data['days_for_shipment_scheduled'] = value
            
            message = """
✅ Excellent!

**Feature 3: shipping_efficiency**
This is the overall shipping performance score (0.0 to 1.0).
- Higher values = better efficiency = lower risk
- Lower values = poor efficiency = higher risk

Please enter **shipping_efficiency** (e.g., 0.75, 0.85, 0.95):
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            return AWAITING_FEATURE_3
        except ValueError:
            await update.message.reply_text("Please enter a valid number (e.g., 4)")
            return AWAITING_FEATURE_2
    except Exception as e:
        logger.error(f"Error in predict_feature_2: {e}")
        await update.message.reply_text("Error processing input. Please try again.")
        return ConversationHandler.END


async def predict_feature_3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        try:
            value = float(user_input)
            context.user_data['shipping_efficiency'] = value
            
            feat_ratio = context.user_data['feat_ratio']
            days_scheduled = context.user_data['days_for_shipment_scheduled']
            efficiency = context.user_data['shipping_efficiency']
            
            feature_vector = np.array([[feat_ratio, days_scheduled, efficiency, 0, 0, 0, 0]])
            
            if model is not None:
                prediction = model.predict(feature_vector)[0]
                proba = model.predict_proba(feature_vector)[0]
                confidence = max(proba) * 100
                
                prediction_class = "⚠️ LATE DELIVERY RISK" if prediction == 1.0 else "✅ ON-TIME DELIVERY"
                risk_emoji = "🔴" if prediction == 1.0 else "🟢"
                
                result_message = f"""
{risk_emoji} **Prediction Result**

**Predicted Outcome:** {prediction_class}
**Confidence:** {confidence:.2f}%

**Input Values:**
• feat_ratio: {feat_ratio}
• days_for_shipment_scheduled: {days_scheduled}
• shipping_efficiency: {efficiency}

**Recommendation:**
"""
                if prediction == 1.0:
                    result_message += """
⚠️ HIGH RISK - Immediate action required!
• Escalate to expedited shipping
• Alert operations manager
• Proactive customer communication
• Consider alternate carriers/routes
"""
                else:
                    result_message += """
✅ LOW RISK - Standard processing
• Continue with normal workflow
• Monitor for any changes
• Maintain current schedule
"""
                
                await update.message.reply_text(result_message, parse_mode='Markdown')
            else:
                await update.message.reply_text("Model not available. Cannot make prediction.")
            
            return ConversationHandler.END
        except ValueError:
            await update.message.reply_text("Please enter a valid number (e.g., 0.85)")
            return AWAITING_FEATURE_3
    except Exception as e:
        logger.error(f"Error in predict_feature_3: {e}")
        await update.message.reply_text("Error making prediction. Please try /predict again.")
        return ConversationHandler.END


async def predict_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("Prediction cancelled. Use /predict to start again.")
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Error in predict_cancel: {e}")
        return ConversationHandler.END


async def insights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            await update.message.reply_text("AI insights unavailable: API key not configured.")
            return
        
        await update.message.reply_text("🤔 Analyzing dataset and generating insights... Please wait.")
        
        total_records = len(df)
        pred_col = 'predictions' if 'predictions' in df.columns else 'late_delivery_risk'
        
        if pred_col in df.columns:
            late_count = (df[pred_col] == 1.0).sum()
            late_pct = (late_count / total_records * 100) if total_records > 0 else 0
        else:
            late_count = 103400
            late_pct = 57.28
        
        context_data = f"""
Dataset: DataCo Global Supply Chain (180,519 orders)
Model: XGBoost Classifier with 97.45% accuracy
Late Delivery Rate: {late_pct:.2f}% ({late_count:,} orders)
Top Predictor: feat_ratio (99.97% importance) - ratio of actual vs scheduled shipping time
Key Finding: Schedule overruns are the dominant factor in late deliveries
Business Context: Operations managers need to prioritize expedited handling and carrier selection
"""
        
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a supply chain analytics expert. Based on this data, provide 2-3 paragraphs of actionable business insights for operations managers:

{context_data}

Focus on: 1) What the data reveals about operational challenges, 2) Specific actions to reduce late delivery risk, 3) ROI opportunities from better prediction."""
                }
            ]
        )
        
        insight_text = message.content[0].text
        
        response = f"""
🧠 **AI-Powered Business Insights**

{insight_text}

---
💡 Generated using Claude AI based on your dataset statistics.
"""
        
        if len(response) > 4096:
            response = response[:4090] + "..."
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in insights: {e}")
        await update.message.reply_text("Sorry, couldn't generate insights. Please try again later.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        help_message = """
📚 **Available Commands**

**/start** - Welcome message and bot introduction

**/stats** - View dataset statistics and model performance metrics

**/top_features** - See the 7 most important features for predicting late deliveries with business explanations

**/hypotheses** - Review validated business hypotheses from the analysis

**/predict** - Interactive prediction tool - answer questions about shipping features to get a late delivery risk prediction

**/insights** - Get AI-generated business insights powered by Claude

**/help** - Show this help message

**About This Bot:**
I analyze 180k+ supply chain orders to predict late delivery risks with 97.45% accuracy, helping operations managers make proactive decisions.

**Need Support?**
Use /start to see the full introduction or try any command above!
"""
        await update.message.reply_text(help_message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in help: {e}")
        await update.message.reply_text("Sorry, couldn't display help. Please try again.")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    try:
        if update and update.message:
            await update.message.reply_text(
                "An unexpected error occurred. Please try again or use /help for assistance."
            )
    except Exception as e:
        logger.error(f"Error in error_handler: {e}")


def main():
    try:
        load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    application = ApplicationBuilder().token(token).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("top_features", top_features))
    application.add_handler(CommandHandler("hypotheses", hypotheses))
    application.add_handler(CommandHandler("insights", insights))
    application.add_handler(CommandHandler("help", help_command))
    
    predict_handler = ConversationHandler(
        entry_points=[CommandHandler("predict", predict_start)],
        states={
            AWAITING_FEATURE_1: [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_feature_1)],
            AWAITING_FEATURE_2: [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_feature_2)],
            AWAITING_FEATURE_3: [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_feature_3)],
        },
        fallbacks=[CommandHandler("cancel", predict_cancel)],
    )
    application.add_handler(predict_handler)
    
    application.add_error_handler(error_handler)
    
    logger.info("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()