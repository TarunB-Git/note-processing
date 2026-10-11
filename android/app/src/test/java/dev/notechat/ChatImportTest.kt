package dev.notechat

import org.junit.Assert.*
import org.junit.Test

class ChatImportTest {
    @Test fun exportedTranscriptRoundTripsTimestampsSendersAndMultilineText() {
        val messages=listOf(Message(text="First line\nSecond line\n\nA paragraph",created=1791705600000,sender="Alex"),Message(text="example.com",created=1791705660000))
        val exported=ChatExport.render("Inbox",messages,emptyList(),false)
        assertFalse(exported.startsWith("Inbox"))
        val parsed=ChatImport.parse(exported)
        assertTrue(parsed.transcript)
        assertEquals(messages.map { it.text },parsed.entries.map { it.text })
        assertEquals(messages.map { it.created },parsed.entries.map { it.created })
        assertEquals(listOf("Alex","Me"),parsed.entries.map { it.sender })
    }
    @Test fun iosTranscriptSupportsSecondsAmPmAndFormatMarks() {
        val parsed=ChatImport.parse("\u200e[10/11/2026, 9:35:12\u202fPM] Alex: Book the flight\ncontinued\n[10/11/2026, 9:36:12 PM] Sam: Why?",true)
        assertEquals(2,parsed.entries.size)
        assertEquals("Book the flight\ncontinued",parsed.entries[0].text)
        assertEquals(60000,parsed.entries[1].created-parsed.entries[0].created)
    }
    @Test fun ambiguousDatesFollowTheChosenDayMonthOrder() {
        val text="10/11/2026, 09:35 - Alex: A note"
        assertNotEquals(ChatImport.parse(text,true).entries.single().created,ChatImport.parse(text,false).entries.single().created)
    }
    @Test fun plainMarkdownKeepsFencedCodeAndListsTogether() {
        val parsed=ChatImport.parse("# Idea\nBuild a garden\n\n```kotlin\nval one = 1\n\nval two = 2\n```\n\n- [ ] Book flight\n- [ ] Pack",now=1000)
        assertFalse(parsed.transcript)
        assertEquals(3,parsed.entries.size)
        assertTrue(parsed.entries[1].text.contains("\n\n"))
        assertTrue(parsed.entries[2].text.contains("Pack"))
    }
    @Test fun bareDomainsAreLinksWithoutMatchingEmailOrTrailingPunctuation() {
        val links=TextLinks.find("Try example.com, www.example.org/path. https://example.net/a(b) and person@example.edu")
        assertEquals(listOf("example.com","www.example.org/path","https://example.net/a(b)"),links.map { it.text })
        assertEquals("https://example.com",links.first().url)
    }
    @Test fun classifierRecognizesBareBookmarks() {
        val classes=listOf(Category("links","Links","Saved references",listOf("https://example.org")),Category("thoughts","Thoughts","Observations",listOf("The evening feels quiet")))
        val classifier=TextClassifier(null); classifier.configure(classes,emptyList())
        assertEquals("links",classifier.classify(Message(text="example.com")).category)
    }
    @Test fun importUsesCurrentTimeUnlessHistoricalDatesAreExplicitlyEnabled() {
        val entries=ChatImport.parse("01/02/2020, 10:00 - Me: old note\n01/02/2020, 10:01 - Me: another note",true).entries
        val current=ChatImport.messages(entries,importedAt=1234)
        assertTrue(current.all { it.created==1234L && !it.approved })
        assertEquals(listOf("old note","another note"),current.map { it.text })
        assertEquals(entries.map { it.created },ChatImport.messages(entries,true,1234).map { it.created })
    }
    @Test fun partialTranscriptKeepsLeadingTextAndDateLikeCodeLines() {
        val source="Leading note without a date\n\n01/02/2020, 10:00 - Me: code example\n```\n01/02/2020, 11:00 - Data: literal\n```"
        val imported=ChatImport.parse(source)
        assertEquals(2,imported.entries.size)
        assertEquals("Leading note without a date",imported.entries.first().text)
        assertTrue(imported.entries.last().text.contains("Data: literal"))
    }
}
