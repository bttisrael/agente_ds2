import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
    ContextTypes
)
import anthropic

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

ASKING_FEAT_RATIO, ASKING_DAYS_SCHEDULED, ASKING_AGGRESSIVE = range(3)

df = None
model = None
feature_importance = {
    'feat_ratio': 0.9904,
    'days_for_shipment_scheduled': 0.0013,
    'aggressive_schedule': 0.0012,
    'sales_per_scheduled_day': 0.0012,
    'feat_interact': 0.0011,
    'log_sales_per_customer': 0.0011,
    'feat_diff': 0.0010
}

hypotheses = [
    "Orders with lower days_for_shipment_scheduled tend to have higher late_delivery_risk",
    "Orders where days_for_shipping_real exceeds days_for_shipment_scheduled tend to have higher late_delivery_risk",
    "Orders with specific type values tend to have higher late_delivery_risk",
    "Orders from certain department_name categories tend to have higher late_delivery_risk",
    "Orders from specific category_name groups tend to have higher late_delivery_risk"
]


def load_data():
    global df, model
    try:
        df = pd.read_parquet('df4_predictions.parquet')
        logger.info(f"Loaded dataframe with {len(df)} rows")
        
        with open('final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Loaded model successfully")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        welcome_msg = (
            "🚀 *Welcome to the Supply Chain Data Science Assistant!*\n\n"
            "I help you understand late delivery predictions for DataCo Global's supply chain operations.\n\n"
            "📊 *Dataset:* 180,519 orders\n"
            "🤖 *Model:* XGBoost Classifier\n"
            "🎯 *Accuracy:* 97.45%\n\n"
            "*Available Commands:*\n"
            "/stats - Dataset and model summary\n"
            "/top_features - Most important prediction features\n"
            "/hypotheses - Validated business insights\n"
            "/predict - Predict late delivery risk\n"
            "/insights - Get AI-powered business insights\n"
            "/help - Show all commands\n\n"
            "Let's optimize your supply chain! 📦✨"
        )
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in start: {e}")
        await update.message.reply_text("Sorry, an error occurred. Please try again.")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        total_records = len(df)
        model_name = "XGBoost Classifier"
        accuracy = 0.9745
        
        pred_counts = {'1.0': 103400, '0.0': 77119}
        total_preds = sum(pred_counts.values())
        
        pct_late = (pred_counts['1.0'] / total_preds) * 100
        pct_ontime = (pred_counts['0.0'] / total_preds) * 100
        
        avg_confidence = "N/A"
        if 'prediction_confidence' in df.columns:
            avg_confidence = f"{df['prediction_confidence'].mean():.2%}"
        
        stats_msg = (
            "📊 *Dataset & Model Summary*\n\n"
            f"📦 *Total Records:* {total_records:,}\n"
            f"🤖 *Model:* {model_name}\n"
            f"🎯 *Accuracy:* {accuracy:.2%}\n\n"
            "*Prediction Distribution:*\n"
            f"🔴 Late Deliveries (1): {pred_counts['1.0']:,} ({pct_late:.1f}%)\n"
            f"🟢 On-Time Deliveries (0): {pred_counts['0.0']:,} ({pct_ontime:.1f}%)\n\n"
            f"💯 *Avg Confidence:* {avg_confidence}\n\n"
            "The model predicts that ~57% of shipments are at risk of late delivery."
        )
        
        await update.message.reply_text(stats_msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in stats: {e}")
        await update.message.reply_text("Sorry, couldn't retrieve stats. Please try again.")


async def top_features(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        features_msg = "🏆 *Top 7 Most Important Features*\n\n"
        
        feature_explanations = {
            'feat_ratio': "📐 *Feat Ratio (99.04%)*\nRatio of actual to scheduled shipping days. The single most critical predictor - shows if shipments are running behind schedule.",
            
            'days_for_shipment_scheduled': "📅 *Days Scheduled (0.13%)*\nPlanned shipping timeframe. Tighter schedules correlate with higher risk of delays.",
            
            'aggressive_schedule': "⚡ *Aggressive Schedule (0.12%)*\nIndicates unrealistically tight delivery windows that are hard to meet.",
            
            'sales_per_scheduled_day': "💰 *Sales per Scheduled Day (0.12%)*\nRevenue intensity per day allocated. Higher values suggest rushed, high-value orders.",
            
            'feat_interact': "🔄 *Feature Interaction (0.11%)*\nCombined effect of multiple scheduling variables working together.",
            
            'log_sales_per_customer': "👤 *Log Sales per Customer (0.11%)*\nTransformed customer value metric. High-value customers may have different risk profiles.",
            
            'feat_diff': "📊 *Feature Difference (0.10%)*\nDifference between actual and scheduled days - direct measure of schedule deviation."
        }
        
        for i, (feat, importance) in enumerate(feature_importance.items(), 1):
            if feat in feature_explanations:
                features_msg += f"{i}. {feature_explanations[feat]}\n\n"
        
        features_msg += "💡 *Key Insight:* feat_ratio dominates with 99% importance - focus on schedule adherence!"
        
        await update.message.reply_text(features_msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in top_features: {e}")
        await update.message.reply_text("Sorry, couldn't retrieve features. Please try again.")


async def show_hypotheses(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        hyp_msg = "✅ *Validated Business Hypotheses*\n\n"
        hyp_msg += "All hypotheses below were tested and confirmed TRUE:\n\n"
        
        explanations = [
            "Tight delivery windows increase risk. *Action:* Add buffer time for critical orders.",
            
            "When actual shipping exceeds planned days, delays are likely. *Action:* Monitor early and escalate.",
            
            "Payment methods (DEBIT, TRANSFER, etc.) correlate with delivery performance. *Action:* Segment by type.",
            
            "Certain product departments have systemic delay issues. *Action:* Audit high-risk departments.",
            
            "Product categories like electronics or furniture show distinct risk patterns. *Action:* Category-specific routing."
        ]
        
        for i, (hyp, exp) in enumerate(zip(hypotheses, explanations), 1):
            hyp_msg += f"{i}. *{hyp.split('tend to')[0].strip()}*\n   → {exp}\n\n"
        
        hyp_msg += "🎯 These insights drive our carrier selection and warehouse routing decisions."
        
        await update.message.reply_text(hyp_msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in show_hypotheses: {e}")
        await update.message.reply_text("Sorry, couldn't retrieve hypotheses. Please try again.")


async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data.clear()
        
        msg = (
            "🔮 *Late Delivery Risk Prediction*\n\n"
            "I'll ask you for the top 3 most important features.\n\n"
            "Let's start!\n\n"
            "📐 *Question 1/3:* What is the *feat_ratio*?\n"
            "(Ratio of actual to scheduled shipping days, e.g., 1.0 = on time, 1.5 = 50% over)"
        )
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        return ASKING_FEAT_RATIO
        
    except Exception as e:
        logger.error(f"Error in predict_start: {e}")
        await update.message.reply_text("Sorry, couldn't start prediction. Please try again.")
        return ConversationHandler.END


async def predict_feat_ratio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        
        try:
            feat_ratio = float(user_input)
            if feat_ratio < 0:
                await update.message.reply_text("Please enter a positive number. Try again:")
                return ASKING_FEAT_RATIO
                
            context.user_data['feat_ratio'] = feat_ratio
            
            msg = (
                "✅ Got it!\n\n"
                "📅 *Question 2/3:* What is *days_for_shipment_scheduled*?\n"
                "(Planned shipping days, typically 2-4 days)"
            )
            await update.message.reply_text(msg, parse_mode='Markdown')
            return ASKING_DAYS_SCHEDULED
            
        except ValueError:
            await update.message.reply_text("❌ Invalid input. Please enter a numeric value (e.g., 1.0, 1.5):")
            return ASKING_FEAT_RATIO
            
    except Exception as e:
        logger.error(f"Error in predict_feat_ratio: {e}")
        await update.message.reply_text("Error processing input. Please try /predict again.")
        return ConversationHandler.END


async def predict_days_scheduled(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        
        try:
            days_scheduled = float(user_input)
            if days_scheduled < 0:
                await update.message.reply_text("Please enter a positive number. Try again:")
                return ASKING_DAYS_SCHEDULED
                
            context.user_data['days_for_shipment_scheduled'] = days_scheduled
            
            msg = (
                "✅ Great!\n\n"
                "⚡ *Question 3/3:* Is this an *aggressive_schedule*?\n"
                "(Enter 1 for YES or 0 for NO)"
            )
            await update.message.reply_text(msg, parse_mode='Markdown')
            return ASKING_AGGRESSIVE
            
        except ValueError:
            await update.message.reply_text("❌ Invalid input. Please enter a numeric value (e.g., 2, 3, 4):")
            return ASKING_DAYS_SCHEDULED
            
    except Exception as e:
        logger.error(f"Error in predict_days_scheduled: {e}")
        await update.message.reply_text("Error processing input. Please try /predict again.")
        return ConversationHandler.END


async def predict_aggressive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        
        try:
            aggressive = float(user_input)
            if aggressive not in [0, 1]:
                await update.message.reply_text("Please enter 0 (NO) or 1 (YES). Try again:")
                return ASKING_AGGRESSIVE
                
            context.user_data['aggressive_schedule'] = aggressive
            
            feat_ratio = context.user_data['feat_ratio']
            days_scheduled = context.user_data['days_for_shipment_scheduled']
            
            input_features = pd.DataFrame([{
                'feat_ratio': feat_ratio,
                'days_for_shipment_scheduled': days_scheduled,
                'aggressive_schedule': aggressive,
                'sales_per_scheduled_day': 100.0,
                'feat_interact': feat_ratio * days_scheduled,
                'log_sales_per_customer': 5.0,
                'feat_diff': (feat_ratio - 1.0) * days_scheduled
            }])
            
            prediction = model.predict(input_features)[0]
            
            try:
                proba = model.predict_proba(input_features)[0]
                confidence = max(proba) * 100
            except:
                confidence = None
            
            result_msg = "🎯 *Prediction Result*\n\n"
            result_msg += f"📥 *Input Values:*\n"
            result_msg += f"  • Feat Ratio: {feat_ratio}\n"
            result_msg += f"  • Days Scheduled: {days_scheduled}\n"
            result_msg += f"  • Aggressive Schedule: {'Yes' if aggressive == 1 else 'No'}\n\n"
            
            if prediction == 1.0:
                result_msg += "🔴 *PREDICTION: LATE DELIVERY RISK*\n\n"
                result_msg += "⚠️ This shipment is at HIGH RISK of late delivery.\n\n"
                result_msg += "*Recommended Actions:*\n"
                result_msg += "• Expedite shipping\n"
                result_msg += "• Alert customer proactively\n"
                result_msg += "• Consider premium carrier\n"
                result_msg += "• Flag for warehouse priority\n"
            else:
                result_msg += "🟢 *PREDICTION: ON-TIME DELIVERY*\n\n"
                result_msg += "✅ This shipment is likely to arrive on time.\n\n"
                result_msg += "*Recommended Actions:*\n"
                result_msg += "• Proceed with standard routing\n"
                result_msg += "• Continue normal monitoring\n"
            
            if confidence:
                result_msg += f"\n💯 *Confidence:* {confidence:.1f}%"
            
            await update.message.reply_text(result_msg, parse_mode='Markdown')
            context.user_data.clear()
            return ConversationHandler.END
            
        except ValueError:
            await update.message.reply_text("❌ Invalid input. Please enter 0 or 1:")
            return ASKING_AGGRESSIVE
            
    except Exception as e:
        logger.error(f"Error in predict_aggressive: {e}")
        await update.message.reply_text("Error making prediction. Please try /predict again.")
        return ConversationHandler.END


async def cancel_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data.clear()
        await update.message.reply_text("Prediction cancelled. Use /predict to start over.")
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Error in cancel_predict: {e}")
        return ConversationHandler.END


async def insights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if not api_key:
            await update.message.reply_text(
                "⚠️ Insights feature requires ANTHROPIC_API_KEY environment variable.\n"
                "Please configure it to use AI-powered insights."
            )
            return
        
        await update.message.reply_text("🤔 Analyzing data and generating insights... Please wait.")
        
        stats_context = f"""
Dataset: DataCo Global Supply Chain (180,519 orders)
Model: XGBoost Classifier with 97.45% accuracy
Target: Late delivery risk prediction

Key Statistics:
- Late deliveries predicted: 103,400 (57.3%)
- On-time deliveries predicted: 77,119 (42.7%)

Top Feature Importance:
1. feat_ratio (99.04%) - ratio of actual to scheduled shipping days
2. days_for_shipment_scheduled (0.13%)
3. aggressive_schedule (0.12%)

Validated Hypotheses:
- Tighter schedules increase late delivery risk
- When actual shipping exceeds scheduled days, delays are very likely
- Certain payment types and product categories show higher risk

Business Context:
Operations managers use these predictions to:
- Route orders through optimal warehouses
- Select appropriate carriers
- Proactively communicate with customers
- Prioritize expedited handling for at-risk shipments
"""
        
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a supply chain analytics expert. Based on this ML project data:

{stats_context}

Provide a 2-3 paragraph business insight focusing on:
1. What the 99% importance of feat_ratio tells us about delivery operations
2. Actionable recommendations for reducing late deliveries
3. How operations managers should prioritize their interventions

Keep it practical, business-focused, and under 500 words."""
                }
            ]
        )
        
        insight_text = message.content[0].text
        
        response_msg = "💡 *AI-Powered Business Insights*\n\n"
        response_msg += insight_text
        response_msg += "\n\n_Generated by Claude AI_"
        
        if len(response_msg) > 4096:
            response_msg = response_msg[:4090] + "..."
        
        await update.message.reply_text(response_msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in insights: {e}")
        await update.message.reply_text(
            "Sorry, couldn't generate insights. Please check your ANTHROPIC_API_KEY and try again."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        help_msg = (
            "🤖 *Supply Chain Assistant - Help*\n\n"
            "*Available Commands:*\n\n"
            "/start - Welcome message and bot overview\n\n"
            "/stats - View dataset summary, model accuracy, and prediction distribution\n\n"
            "/top_features - See the 7 most important features with business explanations\n\n"
            "/hypotheses - List of validated business insights from the analysis\n\n"
            "/predict - Interactive prediction tool - answer 3 questions to get a delivery risk prediction\n\n"
            "/insights - Get AI-powered business insights using Claude API\n\n"
            "/help - Show this help message\n\n"
            "📊 *About the Model:*\n"
            "XGBoost classifier trained on 180K+ supply chain orders to predict late delivery risk.\n\n"
            "❓ Questions? Just try a command and follow the prompts!"
        )
        
        await update.message.reply_text(help_msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in help_command: {e}")
        await update.message.reply_text("Sorry, couldn't display help. Please try again.")


def main():
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        sys.exit(1)
    
    load_data()
    
    application = ApplicationBuilder().token(token).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("top_features", top_features))
    application.add_handler(CommandHandler("hypotheses", show_hypotheses))
    application.add_handler(CommandHandler("insights", insights))
    application.add_handler(CommandHandler("help", help_command))
    
    predict_handler = ConversationHandler(
        entry_points=[CommandHandler("predict", predict_start)],
        states={
            ASKING_FEAT_RATIO: [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_feat_ratio)],
            ASKING_DAYS_SCHEDULED: [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_days_scheduled)],
            ASKING_AGGRESSIVE: [MessageHandler(filters.TEXT & ~filters.COMMAND, predict_aggressive)],
        },
        fallbacks=[CommandHandler("cancel", cancel_predict)],
    )
    
    application.add_handler(predict_handler)
    
    logger.info("Bot started successfully")
    application.run_polling()


if __name__ == "__main__":
    main()